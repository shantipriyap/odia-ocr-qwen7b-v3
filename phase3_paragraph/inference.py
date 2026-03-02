#!/usr/bin/env python3
"""
Odia OCR — Phase 3 Inference Script
=====================================
Runs OCR inference on one or more images using the fine-tuned model from:
  https://huggingface.co/shantipriya/odia-ocr-qwen7b-phase3

Usage:
  python inference.py --image path/to/image.jpg
  python inference.py --image img1.jpg img2.png --device cuda:0
  python inference.py --image img.jpg --base-only    # use Qwen base, no LoRA

Requires:
  pip install transformers torch peft pillow accelerate qwen-vl-utils
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_REPO    = "shantipriya/odia-ocr-qwen7b-phase3"
HF_TOKEN     = ""          # set via --token or HF_TOKEN env var

MAX_NEW_TOKENS = 1024
MAX_IMG_SIZE   = 768
DTYPE          = torch.bfloat16

OCR_PROMPT = (
    "Extract all Odia text from this image exactly as written, "
    "preserving line order and paragraph structure. "
    "Return only the Odia text, nothing else."
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_image(path: str, max_size: int = MAX_IMG_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def run_ocr(image: Image.Image, model, processor, device: str) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  OCR_PROMPT},
    ]}]
    text_prompt = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    output = processor.batch_decode(
        gen[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return output


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import os
    parser = argparse.ArgumentParser(description="Odia OCR inference")
    parser.add_argument("--image",     nargs="+", required=True, help="Image path(s)")
    parser.add_argument("--device",    default="cuda:0",         help="cuda:0 / cpu")
    parser.add_argument("--token",     default="",               help="HuggingFace token")
    parser.add_argument("--base-only", action="store_true",      help="Skip LoRA, use base model")
    parser.add_argument("--lora-repo", default=LORA_REPO,        help="HF LoRA repo ID")
    parser.add_argument("--local-adapter", default="",           help="Path to local adapter dir")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN", "")
    device = args.device

    # Check CUDA availability
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"  Odia OCR — Phase 3 Inference")
    print(f"  Device   : {device}")
    print(f"  Base     : {BASE_MODEL}")
    if not args.base_only:
        src = args.local_adapter if args.local_adapter else args.lora_repo
        print(f"  Adapter  : {src}")
    print(f"{'='*60}\n")

    # Load processor
    from transformers import AutoProcessor
    print("[1/3] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=token or None,
    )

    # Load base model
    from transformers import Qwen2_5_VLForConditionalGeneration
    print("[2/3] Loading base model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=None,
        attn_implementation="flash_attention_2",
        token=token or None,
    ).to(device)

    # Apply LoRA adapter
    if not args.base_only:
        from peft import PeftModel
        adapter_src = args.local_adapter if args.local_adapter else args.lora_repo
        print(f"[3/3] Loading LoRA adapter from {adapter_src} ...")
        try:
            model = PeftModel.from_pretrained(
                model,
                adapter_src,
                token=token or None,
            )
            print("      Adapter loaded ✓")
        except Exception as e:
            print(f"[WARN] Could not load adapter: {e}")
            print("       Running with base model only ...")
    else:
        print("[3/3] Skipping adapter (--base-only)")

    model = model.eval()
    print("\nReady! Processing images ...\n")

    # Run inference on each image
    for img_path in args.image:
        p = Path(img_path)
        if not p.exists():
            print(f"[ERROR] File not found: {img_path}")
            continue

        print(f"── {p.name} {'─'*(50 - len(p.name))}")
        image = load_image(str(p))
        result = run_ocr(image, model, processor, device)
        print(result)
        print()


if __name__ == "__main__":
    main()
