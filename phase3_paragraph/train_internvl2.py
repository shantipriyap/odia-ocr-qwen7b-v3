#!/usr/bin/env python3
"""
Odia OCR — InternVL2-8B Fine-tuning on GPU 1
=============================================
Runs on CUDA:1 independently alongside the Qwen2.5-VL training on CUDA:0.

Launch:
  CUDA_VISIBLE_DEVICES=1 python train_internvl2.py

Model  : OpenGVLab/InternVL2-8B
Output : /root/phase3_paragraph/output_internvl2
HF push: shantipriya/odia-ocr-internvl2-8b  (every SAVE_STEPS)
"""

import os, math, re
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # pin to GPU 1 before any torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL   = "OpenGVLab/InternVL2-8B"
HF_REPO_ID   = "shantipriya/odia-ocr-internvl2-8b"
HF_TOKEN     = os.getenv("HF_TOKEN", "")

DATASET_SYNTHETIC = "OdiaGenAIOCR/synthetic_data"
DATASET_WORD      = "shantipriya/odia-ocr-merged"
WORD_SAMPLE_SIZE  = 2700

OUTPUT_DIR    = "/root/phase3_paragraph/output_internvl2"
MAX_STEPS     = 3000
SAVE_STEPS    = 100
WARMUP_STEPS  = 100
LEARNING_RATE = 5e-5
BATCH_SIZE    = 2
GRAD_ACCUM    = 16      # eff. batch = 32
SEED          = 42

IMG_SIZE      = 448     # InternVL2 native patch size
MAX_TILES     = 6       # dynamic resolution tiles (6 × 448 = ~1.1k visual tokens)
MAX_NEW_TOKENS = 512

LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
# LoRA targets — language model layers inside InternVL2
LORA_TARGETS = [
    "wqkv", "wo",        # InternLM2 attention (fused QKV + output)
    "w1", "w2", "w3",    # InternLM2 MLP (gate / down / up)
]

DTYPE  = torch.bfloat16
DEVICE = torch.device("cuda:0")  # after CUDA_VISIBLE_DEVICES=1, cuda:0 == physical GPU 1

OCR_PROMPT = (
    "<image>\nExtract all Odia text from this image exactly as written, "
    "preserving line order and paragraph structure. "
    "Return only the Odia text, nothing else."
)

print(f"\n{'='*65}")
print(f"  Odia OCR — InternVL2-8B Fine-tuning")
print(f"  Device   : {DEVICE}  (physical GPU 1)")
print(f"  Base     : {BASE_MODEL}")
print(f"  HF push  : {HF_REPO_ID}  (every {SAVE_STEPS} steps)")
print(f"  Output   : {OUTPUT_DIR}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING  (InternVL2 dynamic tile approach)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def dynamic_preprocess(image: Image.Image, max_num: int = MAX_TILES,
                        input_size: int = IMG_SIZE) -> List[Image.Image]:
    """Split image into dynamic tiles (InternVL2 style)."""
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # find best (cols, rows) within max_num tiles
    best_ratio, best = float("inf"), (1, 1)
    for n in range(1, max_num + 1):
        for c in range(1, n + 1):
            r = n // c
            if c * r > max_num:
                continue
            ratio = abs(aspect - c / r)
            if ratio < best_ratio:
                best_ratio, best = ratio, (c, r)

    cols, rows = best
    target_w = input_size * cols
    target_h = input_size * rows
    resized = image.resize((target_w, target_h), Image.BICUBIC)

    tiles = []
    for row in range(rows):
        for col in range(cols):
            box = (col * input_size, row * input_size,
                   (col + 1) * input_size, (row + 1) * input_size)
            tiles.append(resized.crop(box))

    # always add a thumbnail of the full image as the last tile
    thumbnail = image.resize((input_size, input_size), Image.BICUBIC)
    tiles.append(thumbnail)
    return tiles


def encode_image(image: Image.Image) -> torch.Tensor:
    """Returns shape [N_tiles, 3, IMG_SIZE, IMG_SIZE]."""
    tf = build_transform()
    tiles = dynamic_preprocess(image)
    return torch.stack([tf(t) for t in tiles])   # [N, 3, H, W]


def ensure_pil(img, max_size: int = 768) -> Optional[Image.Image]:
    if img is None:
        return None
    try:
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            from io import BytesIO
            pil = Image.open(BytesIO(img["bytes"])).convert("RGB")
        elif isinstance(img, bytes):
            from io import BytesIO
            pil = Image.open(BytesIO(img)).convert("RGB")
        else:
            return None
        w, h = pil.size
        scale = max_size / max(w, h)
        if scale < 1.0:
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return pil
    except Exception:
        return None


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip()) if isinstance(t, str) else ""


def get_gt(sample: dict) -> str:
    for k in ("text", "ground_truth", "label", "caption", "transcription"):
        if k in sample and isinstance(sample[k], str) and sample[k].strip():
            return sample[k].strip()
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_datasets() -> Any:
    print("[DATA] Loading synthetic_data ...")
    synth = load_dataset(DATASET_SYNTHETIC, split="train", trust_remote_code=True)
    print(f"[DATA]   synthetic_data: {len(synth)} samples")

    print("[DATA] Loading odia-ocr-merged ...")
    word_ds = load_dataset(DATASET_WORD, split="train", trust_remote_code=True)
    if len(word_ds) > WORD_SAMPLE_SIZE:
        word_ds = word_ds.shuffle(seed=SEED).select(range(WORD_SAMPLE_SIZE))
    print(f"[DATA]   odia-ocr-merged: {len(word_ds)} samples")

    combined = concatenate_datasets([synth, word_ds]).shuffle(seed=SEED)
    print(f"[DATA] Combined: {len(combined)} samples\n")
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────────────────────
class InternVL2Collator:
    def __init__(self, tokenizer, ignore_index: int = -100):
        self.tokenizer    = tokenizer
        self.ignore_index = ignore_index
        self.im_start_id  = tokenizer.convert_tokens_to_ids("<img>")
        self.im_end_id    = tokenizer.convert_tokens_to_ids("</img>")
        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        pixel_values_list = []
        input_ids_list    = []
        labels_list       = []
        img_counts        = []

        for sample in batch:
            image = ensure_pil(sample.get("image"))
            gt    = clean_text(get_gt(sample))
            if image is None or not gt:
                continue

            pv = encode_image(image)   # [N_tiles, 3, H, W]
            n_tiles = pv.shape[0]
            img_counts.append(n_tiles)

            # Build prompt with <IMG_CONTEXT> tokens
            num_ctx_tokens = 256 * n_tiles   # 256 tokens per tile (InternVL2 default)
            img_token_str  = "<IMG_CONTEXT>" * num_ctx_tokens
            prompt = f"<img>{img_token_str}</img>\n{OCR_PROMPT.replace('<image>', '').strip()}"

            # Encode prompt + answer
            enc_prompt = self.tokenizer(prompt, add_special_tokens=True,
                                        return_tensors="pt")
            enc_answer = self.tokenizer("\n" + gt + self.tokenizer.eos_token,
                                        add_special_tokens=False, return_tensors="pt")

            input_ids = torch.cat([enc_prompt.input_ids[0],
                                   enc_answer.input_ids[0]], dim=0)
            labels = torch.full_like(input_ids, self.ignore_index)
            labels[enc_prompt.input_ids.shape[1]:] = enc_answer.input_ids[0]

            pixel_values_list.append(pv)
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        if not input_ids_list:
            # return a minimal dummy batch (SafeTrainer will skip it)
            T = self.tokenizer
            dummy_ids = torch.tensor([[T.bos_token_id or 1]], dtype=torch.long)
            return {
                "pixel_values":   torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=DTYPE),
                "input_ids":      dummy_ids,
                "attention_mask": torch.ones_like(dummy_ids),
                "labels":         torch.full_like(dummy_ids, -100),
                "image_flags":    torch.zeros(1, dtype=torch.long),
            }

        # Pad sequences to max length
        max_len = max(x.shape[0] for x in input_ids_list)
        pad_id  = self.tokenizer.pad_token_id or 0

        input_ids_padded = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask   = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
        labels_padded    = torch.full((len(labels_list), max_len), -100, dtype=torch.long)

        for i, (ids, lbs) in enumerate(zip(input_ids_list, labels_list)):
            L = ids.shape[0]
            input_ids_padded[i, :L] = ids
            attention_mask[i, :L]   = 1
            labels_padded[i, :L]    = lbs

        # Flatten tiles across batch: InternVL2 expects pixel_values=[total_tiles, 3, H, W]
        # and image_flags=[total_tiles] (1=real tile, 0=padding)
        max_tiles = max(pv.shape[0] for pv in pixel_values_list)
        B = len(pixel_values_list)
        total_tiles = B * max_tiles
        pv_padded    = torch.zeros(total_tiles, 3, IMG_SIZE, IMG_SIZE, dtype=DTYPE)
        image_flags  = torch.zeros(total_tiles, dtype=torch.long)
        for i, pv in enumerate(pixel_values_list):
            n = pv.shape[0]
            pv_padded[i * max_tiles : i * max_tiles + n]   = pv.to(DTYPE)
            image_flags[i * max_tiles : i * max_tiles + n] = 1

        return {
            "pixel_values":  pv_padded,
            "input_ids":     input_ids_padded,
            "attention_mask": attention_mask,
            "labels":        labels_padded,
            "image_flags":   image_flags,
        }

# ─────────────────────────────────────────────────────────────────────────────
# SAFE TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class SafeTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        return model   # never DataParallel

    def _prepare_inputs(self, inputs):
        return self._prepare_input(inputs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not inputs or inputs.get("input_ids") is None:
            return torch.tensor(0.0, device=DEVICE, requires_grad=False)
        # Skip dummy batches (all labels masked = no valid samples in batch)
        labels = inputs.get("labels")
        if labels is not None and (labels == -100).all():
            return torch.tensor(0.0, device=DEVICE, requires_grad=False)
        return super().training_step(model, inputs, num_items_in_batch)

# ─────────────────────────────────────────────────────────────────────────────
# PUSH CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class PushCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if not HF_TOKEN:
            return
        model = kwargs.get("model")
        if model is None:
            return
        print(f"\n[PUSH] step={step} → pushing to {HF_REPO_ID} ...")
        try:
            model.push_to_hub(
                HF_REPO_ID, token=HF_TOKEN,
                commit_message=f"internvl2 step {step}",
            )
            print(f"[PUSH] ✓ pushed step {step}")
        except Exception as e:
            print(f"[PUSH] failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1/5] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, use_fast=False,
        token=HF_TOKEN or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading model (InternVL2-8B) ...")
    model = AutoModel.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=None,
        low_cpu_mem_usage=False,   # avoid meta-tensor .item() crash in InternVL2 ViT init
        _fast_init=False,          # skip meta-device model construction (transformers 5.x)
        trust_remote_code=True,
        token=HF_TOKEN or None,
    ).to(DEVICE)

    # Freeze vision encoder — only train language model
    if hasattr(model, "vision_model"):
        for p in model.vision_model.parameters():
            p.requires_grad = False
    elif hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False

    print("[3/5] Applying LoRA ...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    print("[4/5] Loading dataset ...")
    dataset = load_datasets()
    collator = InternVL2Collator(tokenizer)

    print("[5/5] Starting training ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        dataloader_num_workers=0,   # 0 = main process only; avoids fork/multiprocessing crash
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
    )

    trainer = SafeTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[PushCallback()],
    )

    trainer.train()

    print("\n[DONE] Saving final model ...")
    model.save_pretrained(OUTPUT_DIR + "/final")
    if HF_TOKEN:
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN,
                          commit_message="internvl2 final (step 3000)")
        print(f"[DONE] Pushed final model to {HF_REPO_ID}")


if __name__ == "__main__":
    main()
