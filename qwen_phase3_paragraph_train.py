#!/usr/bin/env python3
"""
Odia OCR — Phase 3 Paragraph Fine-tuning
==========================================
Base model : Qwen/Qwen2.5-VL-7B-Instruct  (upgrade from 3B)
Phase      : 3 — paragraph + mixed training
Goal       : Teach the model to OCR multi-line / paragraph Odia images

Dataset mix:
  - OdiaGenAIOCR/synthetic_data   : 5,349 synthetic paragraph images  (75 %)
  - shantipriya/odia-ocr-merged   : ~2,700 sampled word crops          (25 %)
  Total: ~8,050 samples

Key changes vs Phase 2 (word-only):
  - Base upgraded to 7B (2× capacity over 3B)
  - MAX_NEW_TOKENS 128 → 1024  (paragraph output)
  - MAX_SEQ_LEN   1024 → 4096  (long context)
  - GT_CHAR_BUDGET ~100 → 2400 chars
  - LoRA targets extended: q/k/v/o + gate/up/down projections
  - Paragraph-aware OCR prompt
  - Fresh LoRA on top of 7B base (no dependency on v2 checkpoints)

Run on server (H100 80GB recommended):
  pip install transformers>=4.45 peft>=0.11 trl>=0.9 torch>=2.3
  HF_TOKEN=hf_xxx python qwen_phase3_paragraph_train.py
"""

import os
import json
import re
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image, ImageFilter
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-VL-7B-Instruct"       # 7B — more capacity for paragraphs
HF_REPO_ID  = "shantipriya/odia-ocr-qwen7b-phase3" # push destination
HF_TOKEN    = os.getenv("HF_TOKEN", "")

DATASET_SYNTHETIC = "OdiaGenAIOCR/synthetic_data"   # 5,349 paragraph images
DATASET_WORD      = "shantipriya/odia-ocr-merged"   # 73k word crops (sample 2700)
WORD_SAMPLE_SIZE  = 2700                            # ~25 % of mix

OUTPUT_DIR    = "./output_phase3_para"
MAX_STEPS     = 3000
SAVE_STEPS    = 300
EVAL_STEPS    = 300
WARMUP_STEPS  = 150
LEARNING_RATE = 5e-5     # lower LR — adapting existing VLM knowledge
BATCH_SIZE    = 1
GRAD_ACCUM    = 8        # effective batch = 8; increase to 16 for more stability
EVAL_SAMPLES  = 100
SEED          = 42

# Long-context paragraph settings
MAX_SEQ_LEN    = 4096    # fits long paragraph GT sequences
MAX_IMG_SIZE   = 768     # 768px → ~729 image tokens (Qwen2.5-VL dynamic tiles)
MAX_NEW_TOKENS = 1024    # paragraph-length output
GT_CHAR_BUDGET = 2400    # pre-truncate GT to avoid collator overflow

# LoRA — fresh adapters on 7B base
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

# Paragraph-aware prompt
OCR_PROMPT = (
    "Extract all Odia text from this image exactly as written, "
    "preserving line order and paragraph structure. "
    "Return only the Odia text, nothing else."
)

DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda"  if torch.cuda.is_available() else "cpu"

print(f"\n{'='*65}")
print(f"  Odia OCR — Phase 3 Paragraph Fine-tuning")
print(f"  Base  : {BASE_MODEL}")
print(f"  LR    : {LEARNING_RATE}  Steps: {MAX_STEPS}  Warmup: {WARMUP_STEPS}")
print(f"  SeqLen: {MAX_SEQ_LEN}  ImgSize: {MAX_IMG_SIZE}  MaxNewTok: {MAX_NEW_TOKENS}")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def ensure_pil(img, max_size: int = MAX_IMG_SIZE) -> Optional[Image.Image]:
    """Convert to RGB PIL and downscale longest edge to max_size."""
    pil = None
    if img is None:
        return None
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, bytes):
        from io import BytesIO
        pil = Image.open(BytesIO(img)).convert("RGB")
    elif isinstance(img, str) and os.path.exists(img):
        pil = Image.open(img).convert("RGB")
    else:
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img).convert("RGB")
        except Exception:
            pass
    if pil is None:
        return None
    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil


def clean_odia_text(text: str) -> str:
    """Strip non-Odia noise from ground-truth labels."""
    if not text:
        return ""
    # Keep Odia Unicode block (0B00–0B7F), spaces, newlines, hyphens, digits
    cleaned = re.sub(r"[^\u0B00-\u0B7F\s\n\-\u0966-\u096F0-9।,।]", "", text)
    return cleaned.strip()


def compute_cer(pred: str, ref: str) -> float:
    """Simple character error rate (edit distance / len(ref))."""
    if not ref:
        return 0.0
    import editdistance
    return editdistance.eval(pred, ref) / len(ref)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — load + normalise + mix
# ─────────────────────────────────────────────────────────────────────────────
def load_and_mix_datasets():
    print("[DATA] Loading OdiaGenAIOCR/synthetic_data  (paragraph images)...")
    synth_ds = load_dataset(DATASET_SYNTHETIC, split="train")
    # Normalise field name: extracted_text → gt_text
    synth_ds = synth_ds.rename_column("extracted_text", "gt_text")
    print(f"       synthetic_data : {len(synth_ds):,} samples")

    print("[DATA] Loading shantipriya/odia-ocr-merged  (word crops)...")
    word_ds = load_dataset(DATASET_WORD, split="train")
    word_ds = word_ds.shuffle(seed=SEED).select(range(min(WORD_SAMPLE_SIZE, len(word_ds))))
    print(f"       word crops (sampled) : {len(word_ds):,} samples")

    mixed = concatenate_datasets([synth_ds, word_ds]).shuffle(seed=SEED)
    print(f"[DATA] Mixed total : {len(mixed):,} samples\n")

    # Split off a small eval set
    split   = mixed.train_test_split(test_size=0.03, seed=SEED)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"[DATA] Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}\n")
    return train_ds, eval_ds


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ParagraphOCRCollator:
    processor: Any
    max_length: int = MAX_SEQ_LEN

    def __call__(self, batch: List[Dict]) -> Dict:
        valid_images: List[Image.Image] = []
        texts_input:  List[str]         = []
        truncated_count = 0

        for sample in batch:
            image = ensure_pil(sample.get("image"))
            gt    = clean_odia_text((sample.get("gt_text") or "").strip())
            if image is None or not gt:
                continue

            # Pre-truncate GT to avoid token overflow
            if len(gt) > GT_CHAR_BUDGET:
                gt = gt[:GT_CHAR_BUDGET]
                truncated_count += 1

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  OCR_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": gt,
                },
            ]
            try:
                full_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                valid_images.append(image)
                texts_input.append(full_text)
            except Exception as e:
                print(f"[WARN] apply_chat_template failed: {e}")
                continue

        if truncated_count:
            print(f"[COLLATOR] Pre-truncated {truncated_count}/{len(batch)} samples")
        if not valid_images:
            return {}

        # Retry with progressively smaller images on IndexError
        for attempt in range(3):
            try:
                inputs = self.processor(
                    text=texts_input,
                    images=valid_images,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                break
            except (IndexError, Exception) as e:
                print(f"[WARN] Processor attempt {attempt+1} failed: {e}. Halving image sizes.")
                valid_images = [
                    img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
                    for img in valid_images
                ]
        else:
            return {}

        # Label masking: train only on assistant (GT) tokens
        # Mask everything up to and including the last <|im_start|>assistant token
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id

        # Qwen2.5 chat: assistant turn starts after token 77091 (<|im_start|>)
        # followed by the word "assistant". Mask up to that boundary.
        try:
            assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            for i in range(labels.shape[0]):
                ids = labels[i].tolist()
                # Find last occurrence of <|im_start|>
                last_im_start = -1
                for j in range(len(ids) - 1, -1, -1):
                    if ids[j] == assistant_token_id:
                        last_im_start = j
                        break
                if last_im_start >= 0:
                    labels[i, : last_im_start + 2] = -100  # +2 to also mask "assistant\n"
                else:
                    labels[i, :] = -100  # safety: mask all if boundary not found
        except Exception as e:
            print(f"[WARN] Label masking fallback: {e}")
            labels[labels == pad_id] = -100

        # Also mask padding
        if pad_id is not None:
            labels[labels == pad_id] = -100

        inputs["labels"] = labels
        return dict(inputs)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(model, processor, dataset, tag: str = "eval") -> Dict[str, float]:
    model.eval()
    preds, refs = [], []
    n = min(EVAL_SAMPLES, len(dataset))

    for idx in tqdm(range(n), desc=f"[EVAL] {tag}"):
        sample = dataset[idx]
        image  = ensure_pil(sample.get("image"))
        gt     = clean_odia_text((sample.get("gt_text") or "").strip())
        if image is None or not gt:
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  OCR_PROMPT},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inp = processor(
            text=[text_prompt], images=[image], return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        generated = gen_ids[:, inp["input_ids"].shape[1]:]
        pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        preds.append(pred)
        refs.append(gt)

    # Compute CER using editdistance (install if needed: pip install editdistance)
    try:
        import editdistance
        cer_vals = [
            editdistance.eval(p, r) / max(len(r), 1)
            for p, r in zip(preds, refs)
        ]
        avg_cer = sum(cer_vals) / len(cer_vals) if cer_vals else 1.0
    except ImportError:
        avg_cer = -1.0
        print("[WARN] editdistance not installed — CER skipped. pip install editdistance")

    print(f"\n[{tag.upper()}] Samples: {len(preds)}  |  Avg CER: {avg_cer:.4f}")

    # Print 3 examples
    for i in range(min(3, len(preds))):
        print(f"  GT  : {refs[i][:120]}")
        print(f"  PRED: {preds[i][:120]}")
        print()

    return {"cer": avg_cer, "n_samples": len(preds)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Datasets
    train_ds, eval_ds = load_and_mix_datasets()

    # 2. Processor
    print(f"[MODEL] Loading processor from {BASE_MODEL} ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Ensure padding side is right for decoder-only
    if processor.tokenizer.padding_side != "right":
        processor.tokenizer.padding_side = "right"

    # 3. Model
    print(f"[MODEL] Loading {BASE_MODEL} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 4. Apply LoRA (fresh adapters — no dependency on Phase 2 checkpoints)
    print(f"[LORA] Applying LoRA r={LORA_R}, alpha={LORA_ALPHA} ...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5. Baseline eval (before training)
    print("\n[EVAL] Baseline (before training)...")
    run_evaluation(model, processor, eval_ds, tag="baseline")

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=50,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,   # LoRA checkpoints — handle manually
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to=[],                   # set to ["wandb"] if you use W&B
        seed=SEED,
    )

    # 7. Train
    collator = ParagraphOCRCollator(processor=processor, max_length=MAX_SEQ_LEN)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print(f"\n[TRAIN] Starting Phase 3 training ...")
    print(f"        Max steps : {MAX_STEPS}")
    print(f"        Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"        ~Est. time on H100 : {MAX_STEPS * BATCH_SIZE * GRAD_ACCUM / 3600:.1f} hrs\n")
    trainer.train()

    # 8. Save final LoRA adapter
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n[SAVE] Final adapter saved to: {final_path}")

    # 9. Final eval
    print("\n[EVAL] Final evaluation (after training)...")
    run_evaluation(model, processor, eval_ds, tag="final")

    # 10. Push to HuggingFace Hub
    if HF_TOKEN:
        print(f"\n[HF] Pushing to {HF_REPO_ID} ...")
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN, commit_message="Phase 3 paragraph fine-tune")
        processor.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"[HF] Pushed to https://huggingface.co/{HF_REPO_ID}")
    else:
        print("\n[HF] HF_TOKEN not set — skipping hub push. Set: export HF_TOKEN=hf_xxx")

    # Save training summary
    summary = {
        "base_model": BASE_MODEL,
        "phase": 3,
        "datasets": {
            "synthetic_para": DATASET_SYNTHETIC,
            "word_crops": DATASET_WORD,
            "word_sample_size": WORD_SAMPLE_SIZE,
        },
        "config": {
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "max_seq_len": MAX_SEQ_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gt_char_budget": GT_CHAR_BUDGET,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Training summary: {OUTPUT_DIR}/training_summary.json")


if __name__ == "__main__":
    main()
