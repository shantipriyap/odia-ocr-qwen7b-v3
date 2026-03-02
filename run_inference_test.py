#!/usr/bin/env python3
"""Quick inference test using HF Serverless Inference API.

Tests shantipriya/odia-ocr-qwen-finetuned_v2 on 5 real samples from
shantipriya/odia-ocr-merged and reports exact-match accuracy.
"""
import base64, io, requests, sys
from datasets import load_dataset
from PIL import Image

TOKEN    = "os.getenv("HF_TOKEN", "")"
API_URL  = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"   # serverless-enabled copy
HEADERS  = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
SYSTEM_PROMPT = "Extract the Odia text from this image. Return only the text, nothing else."

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def predict(img: Image.Image) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(img)}"}},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0,
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code != 200:
        return f"ERROR {resp.status_code}: {resp.text[:200]}"
    return resp.json()["choices"][0]["message"]["content"].strip()

# ── Load samples ───────────────────────────────────────────────────────────────
print("Loading dataset (first 30 rows) ...")
ds = load_dataset("shantipriya/odia-ocr-merged", split="train[:30]")
samples = [(ds[i]["image"], ds[i]["text"]) for i in range(min(5, len(ds)))]

print(f"\n{'='*70}")
print(f"  Inference test  |  model: {MODEL_ID}")
print(f"{'='*70}\n")

hits = 0
for idx, (img, gt) in enumerate(samples, 1):
    if isinstance(img, str):
        try: img = Image.open(img).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Skipped (bad image path): {e}")
            continue
    elif hasattr(img, "convert"):
        img = img.convert("RGB")

    pred = predict(img)
    match = pred.strip() == gt.strip()
    hits += int(match)
    mark = "✅" if match else ("⚠️ " if len(pred) > 0 else "❌")
    print(f"[{idx}] {mark}")
    print(f"     GT  : {gt}")
    print(f"     PRED: {pred}")
    print()

acc = hits / len(samples) * 100
print(f"{'='*70}")
print(f"  Exact-match accuracy: {hits}/{len(samples)} = {acc:.0f}%")
print(f"{'='*70}")
