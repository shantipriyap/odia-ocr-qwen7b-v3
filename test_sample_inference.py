#!/usr/bin/env python3
"""Quick smoke-test: fetch one image from shantipriya/odia-ocr-merged and call HF inference.

Env vars used if set:
- HF_OCR_TOKEN (or HF_TOKEN/HUGGINGFACEHUB_API_TOKEN)
- HF_OCR_MODEL (defaults to Qwen/Qwen2.5-VL-3B-Instruct)
- HF_OCR_ADAPTER (defaults to OdiaGenAIOCR/odia-ocr-qwen-finetuned)
"""
import base64
import io
import os
import random
import sys

import requests
from datasets import load_dataset
from PIL import Image
from typing import Optional

API_URL = "https://router.huggingface.co/v1/chat/completions"
CUSTOM_API_URL = os.environ.get("HF_OCR_ENDPOINT_URL")  # Optional direct endpoint URL
HF_INFERENCE_URL_TMPL = "https://api-inference.huggingface.co/models/{model}"
SPACE_API_URL_TMPL = "https://{owner}-{name}.hf.space/api/predict"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
DEFAULT_SPACE = os.environ.get("HF_OCR_SPACE", "OdiaGenAIOCR/odia-ocr-qwen-finetuned")

def pick_token() -> str:
    for key in ("HF_OCR_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    return ""

def load_one_image():
    ds = load_dataset("shantipriya/odia-ocr-merged", split="train", streaming=False)
    sample = ds[0]
    img = sample.get("image")
    if img is None:
        raise RuntimeError("Sample has no image field")
    if isinstance(img, str):
        img = Image.open(img)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        raise RuntimeError("Unknown image type")
    text = sample.get("text", "")
    return img, text


PARA_DATASET = "OdiaGenAIOCR/Odia-lipi-ocr-data"

# ── Tiling parameters ──────────────────────────────────────────────────────────
# Root cause of poor paragraph OCR:
#  1. The LoRA was fine-tuned on word-level images → model outputs short strings.
#  2. Large page images (1400×2000px) use 500+ visual tokens in Qwen2.5-VL,
#     leaving little room for generation.
# Fix: split the page into horizontal strips (~300-400px each), run OCR per
# strip (matching the training distribution), then concatenate results.
STRIP_HEIGHT_PX   = 400   # pixels per strip
STRIP_OVERLAP_PX  = 40    # overlap to avoid cutting words at strip boundaries
STRIP_MAX_WIDTH   = 1400  # rescale width above this before splitting
STRIP_TARGET_W    = 960   # resize each strip to this width before API call
# ──────────────────────────────────────────────────────────────────────────────


def tile_image_strips(img: Image.Image) -> list:
    """Return a list of (Image, y_start, y_end) strip tuples for *img*.

    The image is optionally downscaled and then cut into horizontal strips of
    STRIP_HEIGHT_PX with STRIP_OVERLAP_PX overlap.  Each strip is resized to
    STRIP_TARGET_W width so that the visual token count stays manageable.
    """
    w, h = img.size
    if w > STRIP_MAX_WIDTH:
        scale = STRIP_MAX_WIDTH / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size

    strips = []
    y = 0
    while y < h:
        y_end = min(y + STRIP_HEIGHT_PX, h)
        strip = img.crop((0, y, w, y_end))
        # Resize strip width for consistent token count
        sw, sh = strip.size
        if sw != STRIP_TARGET_W:
            ratio = STRIP_TARGET_W / sw
            strip = strip.resize((STRIP_TARGET_W, max(1, int(sh * ratio))), Image.LANCZOS)
        strips.append((strip, y, y_end))
        if y_end == h:
            break
        y = y_end - STRIP_OVERLAP_PX
    return strips


def infer_tiled_paragraph(
    token: str,
    space_id: str,
    effective_model: str,
    para_img: Image.Image,
    prompt: str,
) -> str:
    """Run tiled OCR over a full-page paragraph image.

    Splits ``para_img`` into horizontal strips and calls ``call_space``
    (falling back to the HF router) on each strip independently.  The
    per-strip results are joined into a single string.

    Args:
        token:           HF API token.
        space_id:        Gradio Space identifier (owner/name).
        effective_model: Router model ID for fallback.
        para_img:        Full-page / paragraph-level image.
        prompt:          OCR prompt.

    Returns:
        Concatenated OCR text from all strips.
    """
    strips = tile_image_strips(para_img)
    print(f"  Tiling: {para_img.size[0]}×{para_img.size[1]}px → {len(strips)} strips")

    collected = []
    for i, (strip_img, y0, y1) in enumerate(strips):
        strip_b64 = to_b64_png(strip_img)
        # Try Space first
        resp, url = call_space(token, space_id, strip_b64, prompt)
        if resp.status_code == 200:
            try:
                result = resp.json()
                output = result.get("data", [result])
                text = str(output[0]).strip() if output else ""
            except Exception:
                text = resp.text.strip()
        else:
            # Router fallback
            pay = router_payload(strip_b64, effective_model, None, max_tokens=300)
            pay["messages"][0]["content"][0]["text"] = prompt
            rr, _ = call_router(token, pay)
            if rr.status_code == 200:
                choices = rr.json().get("choices") or []
                text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
            else:
                text = ""
        print(f"    Strip {i+1}/{len(strips)} (y={y0}-{y1}): {text[:60]!r}")
        if text:
            collected.append(text)

    return "\n".join(collected)


def fetch_random_paragraph(seed: Optional[int] = None) -> tuple:
    """Pick a random sample from the paragraph-level OCR dataset
    ``OdiaGenAIOCR/Odia-lipi-ocr-data`` and return (image, ground_truth_text).

    Each sample already contains a full page/paragraph image with the
    corresponding multi-line Odia ground truth — no synthetic tiling needed.

    Args:
        seed: Optional random seed for reproducibility.  When None a random
              sample is chosen each run.
    """
    ds = load_dataset(PARA_DATASET, split="train", streaming=False)
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"Dataset {PARA_DATASET} is empty")

    rng = random.Random(seed)
    idx = rng.randint(0, n - 1)
    sample = ds[idx]

    img = sample.get("image")
    text = sample.get("text", "").strip()

    if img is None:
        raise RuntimeError(f"Sample {idx} has no image field")
    if isinstance(img, str):
        img = Image.open(img)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        raise RuntimeError("Unknown image type")

    print(f"  [paragraph] dataset={PARA_DATASET}, sample index={idx}/{n-1}, "
          f"image={img.size[0]}×{img.size[1]} px, "
          f"text chars={len(text)}")
    return img, text

def to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def router_payload(img_b64: str, model_id: str, adapter_id: Optional[str], max_tokens=256, temperature=0.0):
    params = {}
    effective_model = model_id
    if adapter_id:
        params["adapter_id"] = adapter_id
    return {
        "model": effective_model,
        "parameters": params or None,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all Odia text from this image. Return only the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

def call_router(token: str, payload: dict):
    api_url = CUSTOM_API_URL or API_URL
    resp = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    return resp, api_url

def call_inference(token: str, model_id: str, adapter_id: Optional[str], img_b64: str, max_tokens=256, temperature=0.0):
    url = HF_INFERENCE_URL_TMPL.format(model=model_id)
    params = {"max_new_tokens": max_tokens, "temperature": temperature}
    if adapter_id:
        params["adapter_id"] = adapter_id
    payload = {
        "inputs": {
            "image": f"data:image/png;base64,{img_b64}",
            "text": "Extract all Odia text from this image. Return only the text.",
        },
        "parameters": params,
    }
    resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=60)
    return resp


def call_space(token: str, space_id: str, img_b64: str, prompt: str = "Extract all Odia text from this image. Return only the text.") -> tuple:
    """Call a deployed HF Gradio Space via its /api/predict endpoint.

    Tries Gradio v3 (/api/predict) and v4+ (/gradio_api/call/predict) formats.
    Returns (response, url_tried).
    """
    owner_raw, name_raw = space_id.split("/", 1)
    base = f"https://{owner_raw.lower()}-{name_raw.lower()}.hf.space"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Gradio v3 format
    url_v3 = f"{base}/api/predict"
    payload_v3 = {"data": [f"data:image/png;base64,{img_b64}", prompt]}
    resp = requests.post(url_v3, headers=headers, json=payload_v3, timeout=90)
    if resp.status_code != 404:
        return resp, url_v3

    # Gradio v4+ format: POST to /gradio_api/call/{fn}, then GET result
    url_v4 = f"{base}/gradio_api/call/predict"
    resp2 = requests.post(url_v4, headers=headers, json={"data": [f"data:image/png;base64,{img_b64}", prompt]}, timeout=30)
    if resp2.status_code == 200:
        try:
            event_id = resp2.json().get("event_id", "")
            if event_id:
                result_resp = requests.get(f"{url_v4}/{event_id}", headers=headers, timeout=90, stream=True)
                # SSE: read lines until we get data
                for line in result_resp.iter_lines():
                    if line and line.startswith(b"data:"):
                        import json as _json
                        data = _json.loads(line[5:])
                        return type("R", (), {"status_code": 200, "json": lambda d=data: d, "text": str(d)})(), url_v4
        except Exception:
            pass
    return resp2, url_v4

def extract_text_from_inference_json(data):
    if isinstance(data, list) and data and isinstance(data[0], dict):
        gen = data[0].get("generated_text") or data[0].get("generated_texts")
        if gen:
            return gen if isinstance(gen, str) else "\n".join(gen)
    if isinstance(data, dict):
        gen = data.get("generated_text") or data.get("text")
        if gen:
            return gen if isinstance(gen, str) else "\n".join(gen)
    return None

def main():
    token = pick_token()
    if not token:
        print("ERROR: Set HF_OCR_TOKEN / HF_TOKEN / HUGGINGFACEHUB_API_TOKEN")
        sys.exit(1)

    model_id = os.environ.get("HF_OCR_MODEL") or DEFAULT_BASE_MODEL
    adapter_id = os.environ.get("HF_OCR_ADAPTER") or DEFAULT_ADAPTER

    # When the fine-tuned adapter is set, use it as the effective model ID
    # for the HF router (it hosts the merged/adapter model directly).
    effective_model = adapter_id if adapter_id else model_id
    print(f"Using model: {model_id}")
    if adapter_id:
        print(f"Using adapter (effective router model): {adapter_id}")
    space_id = os.environ.get("HF_OCR_SPACE", DEFAULT_SPACE)
    print(f"Using Space: {space_id}")
    print("Loading one image from dataset...")
    img, gold_text = load_one_image()
    print(f"Sample ground truth (truncated): {gold_text[:120]!r}")
    img_b64 = to_b64_png(img)

    print("\n--- Space API call (single word) ---")
    space_resp, space_url = call_space(token, space_id, img_b64)
    print(f"Space status: {space_resp.status_code} (url={space_url})")
    if space_resp.status_code == 200:
        try:
            result = space_resp.json()
            # Gradio wraps response in {"data": [...]}
            output = result.get("data", [result])
            extracted = output[0] if output else str(result)
            print("Space output:", str(extracted)[:400])
        except Exception:
            print("Space raw:", space_resp.text[:400])
    else:
        print("Space error:", space_resp.text[:400])
        print("\n--- Router fallback ---")
        router_resp, used_url = call_router(token, router_payload(img_b64, effective_model, None))
        print(f"Router status: {router_resp.status_code}")
        if router_resp.status_code == 200:
            choices = router_resp.json().get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            print("Router output:", content[:400])
        else:
            print("Router error:", router_resp.text[:300])

    # ── Long paragraph test ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARAGRAPH TEST  (real paragraph from OdiaGenAIOCR/Odia-lipi-ocr-data)")
    print("=" * 60)
    try:
        seed_val = os.environ.get("PARA_SEED")  # set for reproducibility
        para_img, para_gold = fetch_random_paragraph(
            seed=int(seed_val) if seed_val else None
        )
        print(f"Paragraph image size: {para_img.size[0]}×{para_img.size[1]} px")
        print(f"Ground truth ({len(para_gold)} chars):\n  {para_gold[:300]}{'...' if len(para_gold) > 300 else ''}")

        # Save locally so the user can inspect it
        out_path = os.path.join(os.path.dirname(__file__), "inference_samples", "paragraph_test.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        para_img.save(out_path)
        print(f"Saved paragraph image → {out_path}")

        para_prompt = (
            "Extract all the Odia text from this image exactly as it appears, "
            "preserving line breaks. Return only the text, no explanation."
        )

        print("\n--- Paragraph tiled OCR (strips strategy) ---")
        print("  (Splitting full-page image into horizontal strips to match")
        print("   the word-level training distribution of the fine-tuned model.)")
        try:
            extracted = infer_tiled_paragraph(
                token, space_id, effective_model, para_img, para_prompt
            )
            print(f"\nExtracted text ({len(extracted)} chars):")
            print(extracted[:1200] + ("..." if len(extracted) > 1200 else ""))
        except Exception as exc2:
            print(f"Tiled inference error: {exc2}")
    except Exception as exc:
        print(f"Paragraph test failed: {exc}")

if __name__ == "__main__":
    main()
