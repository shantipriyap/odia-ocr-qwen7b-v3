#!/usr/bin/env python3
"""Quick inference test from OdiaGenAIOCR/odia-ocr-qwen-finetuned via HF router."""

import base64, io, requests
from PIL import Image, ImageDraw, ImageFont

TOKEN = "os.getenv("HF_TOKEN", "")"
MODEL = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
API_URL    = "https://api-inference.huggingface.co/models/" + MODEL

# ---------- create a simple test image (white background + label) ----------
def make_test_image():
    img = Image.new("RGB", (320, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Add a visible caption so there is something to "OCR"
    draw.text((10, 20), "Odia OCR Test Image", fill=(30, 30, 30))
    return img

def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- try to fetch a real sample from HF hub (fast, small dataset) ----
def get_sample_image():
    """Try to load one image from OdiaGenAIOCR/Odia-lipi-ocr-data (faster hf_hub path)."""
    try:
        from huggingface_hub import hf_hub_download
        import json, os

        # Download a single shard manifest to find an image file
        print("  Trying hf_hub_download for a sample image...")
        fp = hf_hub_download(
            repo_id="OdiaGenAIOCR/Odia-lipi-ocr-data",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=TOKEN,
        )
        import pandas as pd
        df = pd.read_parquet(fp)
        row = df.iloc[0]
        img_data = row.get("image") or row.get("img")
        text = row.get("text", "")
        if img_data is not None:
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                raise ValueError("Unknown image format")
            return img, text
    except Exception as e:
        print(f"  Couldn't load from dataset ({e}), using synthetic image")
    return make_test_image(), "(synthetic test image)"

def call_router(img_b64):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all Odia text from this image. Return only the text."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    resp = requests.post(
        ROUTER_URL,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    return resp

def call_direct_api(img_b64):
    """Fallback: direct HF Inference API (no vision support for most models, but try)."""
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"inputs": f"data:image/png;base64,{img_b64}"},
        timeout=60,
    )
    return resp

def main():
    print("=" * 65)
    print("  ODIA OCR INFERENCE — OdiaGenAIOCR/odia-ocr-qwen-finetuned")
    print("=" * 65)

    print("\n[1] Loading sample image...")
    img, gold = get_sample_image()
    print(f"    Image size : {img.size[0]}x{img.size[1]} px")
    print(f"    Ground truth: {gold[:120]!r}")
    img_b64 = to_b64(img)

    # Save image locally for inspection
    out = "/tmp/odia_test_img.png"
    img.save(out)
    print(f"    Saved to   : {out}")

    print("\n[2] Calling HF Router → OdiaGenAIOCR/odia-ocr-qwen-finetuned ...")
    resp = call_router(img_b64)
    print(f"    Status     : {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            print(f"\n    EXTRACTED TEXT:\n    {content[:600]}")
        else:
            print(f"    Response   : {str(data)[:400]}")
    else:
        print(f"    Error body : {resp.text[:500]}")
        print("\n[3] Trying direct Inference API as fallback...")
        resp2 = call_direct_api(img_b64)
        print(f"    Status     : {resp2.status_code}")
        print(f"    Response   : {resp2.text[:400]}")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)

if __name__ == "__main__":
    main()
