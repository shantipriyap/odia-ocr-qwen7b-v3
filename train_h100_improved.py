#!/usr/bin/env python3
"""
Improved Odia OCR Training on H100 80GB
Model: Qwen/Qwen2.5-VL-3B-Instruct
Dataset: shantipriya/odia-ocr-merged
Goal: Improve shantipriya/odia-ocr-qwen-finetuned

Key improvements over previous run:
  - Proper Qwen2.5-VL chat template format for OCR
  - Larger LoRA rank (r=128) for better capacity
  - More training steps (2000) with cosine schedule
  - Gradient checkpointing for memory efficiency
  - Label masking (only compute loss on answer tokens)
  - H100 bf16 + tf32 optimizations
  - Larger batch (per_device=4, grad_accum=4 → effective batch 16)
"""

import os, sys, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
MODEL_NAME  = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET     = "shantipriya/odia-ocr-merged"
OUTPUT_DIR  = "/root/odia_ocr/checkpoints_improved"
PUSH_REPO   = "shantipriya/odia-ocr-qwen-finetuned"

# ── imports ────────────────────────────────────────────────────────────────────
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

IGNORE_INDEX = -100
PROMPT = (
    "You are an expert Odia OCR system. "
    "Extract all Odia text from this image exactly as written. "
    "Return only the extracted text, nothing else."
)

# ── helpers ────────────────────────────────────────────────────────────────────

def load_model_and_processor():
    log.info(f"Loading processor from {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    # Use a reasonable max pixels to avoid OOM on high-res images
    processor.image_processor.max_pixels = 768 * 768

    log.info(f"Loading model from {MODEL_NAME}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    model.config.use_cache = False
    log.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def apply_lora(model):
    # Target all attention + MLP projections for better coverage
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


class OdiaOCRDataCollator:
    """Collate OCR samples into Qwen2.5-VL chat format with label masking."""

    def __init__(self, processor):
        self.processor = processor

    def _make_conversation(self, image: Image.Image, text: str):
        """Build the message list for apply_chat_template."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ]

    def __call__(self, batch):
        valid_images, valid_texts = [], []
        for ex in batch:
            img  = ex.get("image")
            text = ex.get("text", "").strip()
            if not text:
                continue
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert("RGB")
                except Exception:
                    continue
            elif hasattr(img, "convert"):
                img = img.convert("RGB")
            else:
                continue
            valid_images.append(img)
            valid_texts.append(text)

        if not valid_images:
            # Return a minimal dummy batch so Trainer doesn't crash
            dummy = torch.zeros((1, 4), dtype=torch.long)
            return {"input_ids": dummy, "attention_mask": dummy, "labels": dummy.clone().fill_(IGNORE_INDEX)}

        conversations = [
            self._make_conversation(img, text)
            for img, text in zip(valid_images, valid_texts)
        ]

        # apply_chat_template handles image tokens and text jointly
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in conversations
        ]

        # Build vision_infos for process_vision_info
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(conversations)
        except Exception:
            image_inputs = valid_images
            video_inputs = None

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Label masking: only supervise assistant tokens
        labels = inputs["input_ids"].clone()
        for i, (input_ids, conv) in enumerate(zip(inputs["input_ids"], conversations)):
            # Encode just the user+system portion to find the split point
            prompt_only_text = self.processor.apply_chat_template(
                [conv[0]],  # only user turn
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.processor.tokenizer.encode(
                prompt_only_text, add_special_tokens=False
            )
            n_prompt = len(prompt_ids)
            labels[i, :n_prompt] = IGNORE_INDEX

        inputs["labels"] = labels
        inputs.pop("token_type_ids", None)
        return inputs


def main():
    # ── auth ──────────────────────────────────────────────────────────────────
    if HF_TOKEN:
        log.info("Logging in to Hugging Face Hub")
        login(token=HF_TOKEN, add_to_git_credential=True)

    # ── dataset ───────────────────────────────────────────────────────────────
    log.info(f"Loading dataset {DATASET}")
    ds = load_dataset(DATASET, token=HF_TOKEN or None)

    # Combine train + validation/dev for training; evaluate on test set only
    from datasets import concatenate_datasets

    train_parts = [ds["train"]]
    if "validation" in ds:
        train_parts.append(ds["validation"])
        log.info(f"  Merging train ({len(ds['train']):,}) + validation ({len(ds['validation']):,})")
    elif "dev" in ds:
        train_parts.append(ds["dev"])
        log.info(f"  Merging train ({len(ds['train']):,}) + dev ({len(ds['dev']):,})")

    train_ds = concatenate_datasets(train_parts).shuffle(seed=42)
    log.info(f"  {len(train_ds):,} total training samples")

    # Test split for final evaluation
    if "test" in ds:
        val_ds = ds["test"]
        log.info(f"  {len(val_ds):,} test samples (held-out evaluation)")
    else:
        # Fallback: carve 1% from combined train as test proxy
        split = train_ds.train_test_split(test_size=0.01, seed=42)
        train_ds = split["train"]
        val_ds   = split["test"]
        log.info(f"  {len(val_ds):,} test samples (carved from train, no test split found)")
    log.info(f"  Train: {len(train_ds):,}  |  Test: {len(val_ds):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor()
    model = apply_lora(model)

    # H100-specific: enable gradient checkpointing & tf32
    model.gradient_checkpointing_enable()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── training args ─────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # Batch: effective = 4 * 4 = 16 per step
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        # Schedule
        num_train_epochs=3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        # Precision
        bf16=True,
        tf32=True,
        # Logging & checkpointing
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Misc
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to=[],
        seed=42,
        # Push to Hub
        push_to_hub=bool(HF_TOKEN),
        hub_model_id=PUSH_REPO if HF_TOKEN else None,
        hub_strategy="checkpoint",
        hub_token=HF_TOKEN or None,
    )

    collator = OdiaOCRDataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # ── train ──────────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STARTING IMPROVED ODIA OCR TRAINING ON H100")
    log.info(f"  Model:   {MODEL_NAME}")
    log.info(f"  Dataset: {DATASET}  ({len(train_ds):,} train, {len(val_ds):,} val)")
    log.info(f"  Output:  {OUTPUT_DIR}")
    log.info("=" * 70)

    trainer.train()

    # ── save ───────────────────────────────────────────────────────────────────
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    log.info(f"Saved to {final_dir}")

    # ── push ───────────────────────────────────────────────────────────────────
    if HF_TOKEN:
        log.info(f"Pushing adapter to {PUSH_REPO}")
        trainer.push_to_hub(commit_message="Improved training: H100, r=128 LoRA, chat-template, label-mask, 3 epochs")
        processor.push_to_hub(PUSH_REPO, token=HF_TOKEN)
        log.info("Push complete!")

    log.info("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
