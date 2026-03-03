#!/usr/bin/env python3
"""
Quick evaluation of InternVL2-8B LoRA checkpoint on Odia OCR.
Loads 5 samples from OdiaGenAIOCR/synthetic_data and runs inference.
Uses CUDA:1 so it doesn't clash with Qwen training on CUDA:0.
"""

import os, io, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL  = "OpenGVLab/InternVL2-8B"
CKPT_DIR    = "/root/phase3_paragraph/output_internvl2/checkpoint-400"
HF_TOKEN    = "YOUR_HF_TOKEN_HERE"
NUM_SAMPLES = 5
IMG_SIZE    = 448

# ─── Image preprocessing (same as training) ──────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(size=IMG_SIZE):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

transform = build_transform()

def load_image(sample):
    """Extract PIL image from dataset sample (handles dict/bytes/PIL)."""
    img_field = sample.get("image") or sample.get("img")
    if img_field is None:
        keys = list(sample.keys())
        raise ValueError(f"No image field found. Keys: {keys}")
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, bytes):
        return Image.open(io.BytesIO(img_field)).convert("RGB")
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    raise ValueError(f"Unknown image type: {type(img_field)}")

def get_text(sample):
    for key in ("extracted_text", "text", "ground_truth", "label", "caption", "transcription"):
        if key in sample and sample[key]:
            return str(sample[key]).strip()
    raise ValueError(f"No text field. Keys: {list(sample.keys())}")

# ─── CER ─────────────────────────────────────────────────────────────────────
def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via dynamic programming."""
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    if n == 0:
        return float(m)
    d = np.zeros((n+1, m+1), dtype=int)
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[n][m] / n

# ─── Load model ──────────────────────────────────────────────────────────────
print("Loading base model InternVL2-8B (no LoRA for pipeline test) ...")
model = AutoModel.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    token=HF_TOKEN,
    device_map={"": "cuda:0"},   # after CUDA_VISIBLE_DEVICES=1, cuda:0 = physical GPU 1
)
model.eval()
print("Base model loaded (baseline eval — no LoRA).")

tokenizer = AutoTokenizer.from_pretrained(
    CKPT_DIR,
    trust_remote_code=True,
    token=HF_TOKEN,
)
print("Model loaded.\n")

# ─── Load dataset samples ─────────────────────────────────────────────────────
print("Loading dataset samples ...")
ds = load_dataset(
    "OdiaGenAIOCR/synthetic_data",
    split="train",
    streaming=True,
    token=HF_TOKEN,
)
samples = []
for s in ds:
    if len(samples) >= NUM_SAMPLES:
        break
    try:
        img = load_image(s)
        txt = get_text(s)
        if txt and len(txt) > 30:   # skip trivially short labels
            samples.append((img, txt))
    except Exception as e:
        pass   # silently skip bad rows

print(f"Got {len(samples)} samples.\n")

# ─── Run inference ────────────────────────────────────────────────────────────
PROMPT = "<image>\nTranscribe all the Odia text from this image exactly as it appears. Output only the transcribed text, nothing else."

results = []
for i, (img, gt) in enumerate(samples, 1):
    pixel_values = transform(img).unsqueeze(0).to(torch.bfloat16).to(model.device)
    try:
        with torch.no_grad():
            pred = model.chat(
                tokenizer,
                pixel_values,
                PROMPT,
                dict(max_new_tokens=512, do_sample=False),
            )
    except Exception as e:
        pred = f"[ERROR: {e}]"

    c = cer(gt, pred)
    results.append((gt, pred, c))

    print(f"{'='*70}")
    print(f"Sample {i}")
    print(f"GT  : {gt[:200]}")
    print(f"PRED: {pred[:200]}")
    print(f"CER : {c:.4f}  |  ACC: {max(0, 1-c)*100:.1f}%")
    print()

# ─── Summary ─────────────────────────────────────────────────────────────────
avg_cer = np.mean([r[2] for r in results])
print(f"{'='*70}")
print(f"Average CER  : {avg_cer:.4f}")
print(f"Average ACC  : {max(0, 1-avg_cer)*100:.1f}%")
print(f"{'='*70}")
