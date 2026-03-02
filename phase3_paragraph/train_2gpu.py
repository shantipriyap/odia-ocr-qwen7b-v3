#!/usr/bin/env python3
"""
Odia OCR — Phase 3 Paragraph Fine-tuning  (2× A100-80GB, split roles)
=======================================================================
Architecture
  GPU 0  — training  (HF Trainer, single-GPU, gradient updates)
  GPU 1  — eval + HF Hub push  (background thread, never touches GPU 0)

Launch:
  python train_2gpu.py

No torchrun / DDP needed. Both GPUs are used simultaneously:
  GPU 0 trains continuously, GPU 1 runs inference for eval and pushes
  adapter snapshots to the Hub every PUSH_STEPS steps.

Effective batch = per_device_batch(4) × grad_accum(8) = 32
MAX_STEPS=3000  → ~1.5-2 hours on a single A100-80GB
"""

import os, re, threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_REPO_ID  = "shantipriya/odia-ocr-qwen7b-phase3"
HF_TOKEN    = os.getenv("HF_TOKEN", "")

DATASET_SYNTHETIC = "OdiaGenAIOCR/synthetic_data"
DATASET_WORD      = "shantipriya/odia-ocr-merged"
WORD_SAMPLE_SIZE  = 2700

OUTPUT_DIR    = "/root/phase3_paragraph/output_2gpu"
MAX_STEPS     = 3000
SAVE_STEPS    = 100
PUSH_STEPS    = 100
WARMUP_STEPS  = 100
LEARNING_RATE = 5e-5
BATCH_SIZE    = 4       # per-device (GPU 0)  — safe with smaller images
GRAD_ACCUM    = 8       # eff. batch = 4 × 8 = 32
EVAL_SAMPLES  = 5       # quick eval on GPU 1 each checkpoint
SEED          = 42

MAX_SEQ_LEN    = 2048
MAX_IMG_SIZE   = 512   # fewer visual tokens: ~335 vs ~750 at 768px
MAX_NEW_TOKENS = 1024
GT_CHAR_BUDGET = 2400

LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

OCR_PROMPT = (
    "Extract all Odia text from this image exactly as written, "
    "preserving line order and paragraph structure. "
    "Return only the Odia text, nothing else."
)

TRAIN_DEVICE = torch.device("cuda:0")
EVAL_DEVICE  = torch.device("cuda:1")
DTYPE        = torch.bfloat16

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print(f"\n{'='*65}")
print(f"  Odia OCR — Phase 3 Paragraph")
print(f"  Train GPU : {TRAIN_DEVICE}   Eval/Push GPU : {EVAL_DEVICE}")
print(f"  Base      : {BASE_MODEL}")
print(f"  Eff. batch: {BATCH_SIZE * GRAD_ACCUM}  Steps: {MAX_STEPS}")
print(f"  HF Hub    : {HF_REPO_ID}  (every {PUSH_STEPS} steps)")
print(f"  Output    : {OUTPUT_DIR}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def ensure_pil(img, max_size: int = MAX_IMG_SIZE) -> Optional[Image.Image]:
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
    if not text:
        return ""
    cleaned = re.sub(r"[^\u0B00-\u0B7F\s\n\-\u0966-\u096F0-9\u0964,]", "", text)
    return cleaned.strip()


def get_gt(sample: dict) -> str:
    for key in ("gt_text", "text", "transcription", "label", "caption", "ocr_text"):
        v = sample.get(key)
        if v:
            return str(v)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_and_mix_datasets():
    print("[DATA] Loading OdiaGenAIOCR/synthetic_data ...")
    synth_ds = load_dataset(DATASET_SYNTHETIC, split="train")
    synth_ds = synth_ds.rename_column("extracted_text", "gt_text")
    print(f"       synthetic_data : {len(synth_ds):,} paragraph samples")

    print("[DATA] Loading shantipriya/odia-ocr-merged ...")
    word_ds = load_dataset(DATASET_WORD, split="train")
    word_ds = word_ds.shuffle(seed=SEED).select(range(min(WORD_SAMPLE_SIZE, len(word_ds))))
    for candidate in ("text", "transcription", "label", "caption", "ocr_text"):
        if candidate in word_ds.column_names:
            word_ds = word_ds.rename_column(candidate, "gt_text")
            break

    mixed = concatenate_datasets([synth_ds, word_ds]).shuffle(seed=SEED)
    split = mixed.train_test_split(test_size=0.03, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[DATA] Train: {len(train_ds):,}  Eval: {len(eval_ds):,}\n")
    return train_ds, eval_ds


# ─────────────────────────────────────────────────────────────────────────────
# COLLATOR  (builds batches for GPU 0)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ParagraphOCRCollator:
    processor: Any
    max_length: int = MAX_SEQ_LEN

    def __call__(self, batch: List[Dict]) -> Dict:
        valid_images: List[Image.Image] = []
        texts_input:  List[str]         = []

        for sample in batch:
            image = ensure_pil(sample.get("image"))
            gt    = clean_odia_text(get_gt(sample).strip())
            if image is None or not gt:
                continue
            if len(gt) > GT_CHAR_BUDGET:
                gt = gt[:GT_CHAR_BUDGET]

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  OCR_PROMPT},
                ]},
                {"role": "assistant", "content": gt},
            ]
            try:
                full_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                valid_images.append(image)
                texts_input.append(full_text)
            except Exception as e:
                print(f"[WARN] template: {e}")
                continue

        if not valid_images:
            return {}

        for attempt in range(3):
            try:
                inputs = self.processor(
                    text=texts_input, images=valid_images,
                    padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                break
            except Exception as e:
                print(f"[WARN] Processor attempt {attempt+1}: {e}. Halving images.")
                valid_images = [
                    img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
                    for img in valid_images
                ]
        else:
            return {}

        # Label masking — supervise only on assistant (GT) tokens
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        try:
            asst_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            for i in range(labels.shape[0]):
                ids = labels[i].tolist()
                last_im = max((j for j, x in enumerate(ids) if x == asst_id), default=-1)
                if last_im >= 0:
                    labels[i, : last_im + 2] = -100
                else:
                    labels[i, :] = -100
        except Exception as e:
            print(f"[WARN] Label mask: {e}")
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return dict(inputs)


# ─────────────────────────────────────────────────────────────────────────────
# EVAL + PUSH  (runs entirely on GPU 1 in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
_eval_lock = threading.Lock()


def run_eval_and_push(
    adapter_path: str,
    processor,
    eval_ds,
    step: int,
    tag: str = "",
    push: bool = True,
    is_base: bool = False,
) -> None:
    """Load adapter (or base model) on GPU 1, run quick eval, push to Hub."""
    with _eval_lock:
        eval_model = None
        base_m = None
        try:
            print(f"\n[GPU1] step={step} tag={tag or 'ckpt'} — loading model ...")
            base_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL, torch_dtype=DTYPE, device_map=None,
                attn_implementation="flash_attention_2",
            ).to(EVAL_DEVICE)

            if not is_base:
                eval_model = PeftModel.from_pretrained(base_m, adapter_path).eval()
            else:
                eval_model = base_m.eval()

            preds, refs = [], []
            for idx in range(min(EVAL_SAMPLES, len(eval_ds))):
                sample = eval_ds[idx]
                image  = ensure_pil(sample.get("image"))
                ref    = clean_odia_text(get_gt(sample).strip())
                if image is None or not ref:
                    continue
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  OCR_PROMPT},
                ]}]
                text_p = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inp = processor(
                    text=[text_p], images=[image], return_tensors="pt"
                ).to(EVAL_DEVICE)
                with torch.no_grad():
                    gen = eval_model.generate(
                        **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                    )
                pred = processor.batch_decode(
                    gen[:, inp["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
                preds.append(pred)
                refs.append(ref)

            if preds:
                try:
                    import editdistance
                    cer = sum(
                        editdistance.eval(p, r) / max(len(r), 1)
                        for p, r in zip(preds, refs)
                    ) / len(preds)
                except ImportError:
                    cer = -1.0
                lbl = f"[GPU1 EVAL step={step}{'/'+tag if tag else ''}]"
                print(f"\n{lbl}  n={len(preds)}  CER={cer:.4f}")
                for i in range(min(2, len(preds))):
                    print(f"  GT  : {refs[i][:120]}")
                    print(f"  PRED: {preds[i][:120]}\n")

            if push and HF_TOKEN and not is_base:
                print(f"[GPU1] Pushing to {HF_REPO_ID} (step {step}) ...")
                try:
                    eval_model.push_to_hub(
                        HF_REPO_ID, token=HF_TOKEN,
                        commit_message=f"phase3 step {step}{' '+tag if tag else ''}",
                    )
                    print(f"[GPU1] Pushed ✓  (step {step})")
                except Exception as e:
                    print(f"[GPU1] Push failed: {e}")

        except Exception as e:
            import traceback
            print(f"[GPU1] ERROR in eval/push: {e}")
            traceback.print_exc()
        finally:
            try:
                del eval_model, base_m
            except Exception:
                pass
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class EvalPushCallback(TrainerCallback):
    """Spawns a GPU-1 eval+push thread on every checkpoint save."""

    def __init__(self, processor, eval_ds):
        self.processor = processor
        self.eval_ds   = eval_ds
        self._threads: List[threading.Thread] = []

    def on_save(self, args, state, control, **kwargs):
        step     = state.global_step
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = args.output_dir
        t = threading.Thread(
            target=run_eval_and_push,
            args=(ckpt_dir, self.processor, self.eval_ds, step),
            kwargs={"push": True},
            daemon=True,
        )
        t.start()
        self._threads.append(t)
        print(f"[GPU1] Eval+push thread started for step {step}")

    def on_train_end(self, args, state, control, **kwargs):
        step = state.global_step
        print(f"\n[GPU1] Waiting for background threads ...")
        for t in self._threads:
            t.join(timeout=900)
        # Final push with full output_dir
        t = threading.Thread(
            target=run_eval_and_push,
            args=(args.output_dir, self.processor, self.eval_ds, step),
            kwargs={"tag": "FINAL", "push": True},
            daemon=False,
        )
        t.start()
        t.join(timeout=900)
        print(f"[GPU1] All done ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SAFE TRAINER  (skips the _signature_columns crash in some transformers vers.)
# ─────────────────────────────────────────────────────────────────────────────
class SafeTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        # Prevent Trainer from wrapping in nn.DataParallel when 2 GPUs are
        # visible. GPU 1 is reserved for the eval/push thread.
        return model

    def _prepare_inputs(self, inputs):
        return self._prepare_input(inputs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not inputs:   # empty batch from collator
            return torch.tensor(0.0, device=TRAIN_DEVICE, requires_grad=False)
        return super().training_step(model, inputs, num_items_in_batch)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, eval_ds = load_and_mix_datasets()

    print("[MODEL] Loading processor ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if processor.tokenizer.padding_side != "right":
        processor.tokenizer.padding_side = "right"

    print(f"[MODEL] Loading {BASE_MODEL} → {TRAIN_DEVICE} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=None,                          # single GPU — let .to() place it
        attn_implementation="flash_attention_2",  # A100 native FA2
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = model.to(TRAIN_DEVICE)

    print(f"[LORA] Applying LoRA r={LORA_R} ...")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()  # required for gradient checkpointing with PEFT
    model.print_trainable_parameters()

    # Baseline eval on GPU 1 — run AFTER model is fully ready
    print("[GPU1] Queuing baseline eval on GPU 1 ...")
    t_base = threading.Thread(
        target=run_eval_and_push,
        args=(BASE_MODEL, processor, eval_ds, 0),
        kwargs={"tag": "baseline", "push": False, "is_base": True},
        daemon=True,
    )
    t_base.start()

    collator = ParagraphOCRCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=25,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,   # saves ~50% activation memory
        report_to="none",
        seed=SEED,
    )

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[EvalPushCallback(processor=processor, eval_ds=eval_ds)],
    )

    print(f"\n[TRAIN] Starting on {TRAIN_DEVICE} — eff.batch={BATCH_SIZE*GRAD_ACCUM}  steps={MAX_STEPS}")
    trainer.train()

    print("[SAVE] Saving final adapter ...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
