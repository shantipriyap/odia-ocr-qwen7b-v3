#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and push merged weights to HuggingFace.

Steps:
  1. Load Qwen2.5-VL-3B-Instruct base model
  2. Apply OdiaGenAIOCR/odia-ocr-qwen-finetuned LoRA adapter
  3. Merge weights with merge_and_unload()
  4. Save merged model locally
  5. Push to OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged on HF
"""
import os, sys, shutil, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from huggingface_hub import HfApi

TOKEN       = os.environ.get("HF_TOKEN", "os.getenv("HF_TOKEN", "")")
BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER     = "shantipriya/odia-ocr-qwen-finetuned_v2"   # v2 LoRA adapter
MERGED_REPO = "OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged"
LOCAL_DIR   = "/root/odia_ocr/merged_model"

print("=" * 70)
print("  MERGE LORA → FULL MODEL & PUSH TO HF")
print(f"  Base   : {BASE_MODEL}")
print(f"  Adapter: {ADAPTER}")
print(f"  Target : {MERGED_REPO}")
print("=" * 70)

# ── 1. Create HF repo ──────────────────────────────────────────────────────────
api = HfApi(token=TOKEN)
try:
    api.create_repo(
        repo_id=MERGED_REPO,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    print(f"\n[1] Repo {MERGED_REPO} ready on HF")
except Exception as e:
    print(f"[1] Repo creation note: {e}")

# ── 2. Load processor ──────────────────────────────────────────────────────────
print(f"\n[2] Loading processor from {BASE_MODEL} ...")
processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    token=TOKEN,
)
print("    Processor loaded ✓")

# ── 3. Load base model ─────────────────────────────────────────────────────────
print(f"\n[3] Loading base model {BASE_MODEL} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=TOKEN,
)
print("    Base model loaded ✓")

# ── 4. Apply adapter ───────────────────────────────────────────────────────────
print(f"\n[4] Loading LoRA adapter from {ADAPTER} ...")
peft_model = PeftModel.from_pretrained(
    model,
    ADAPTER,
    token=TOKEN,
)
print("    Adapter loaded ✓")

# ── 5. Merge & unload ──────────────────────────────────────────────────────────
print("\n[5] Merging LoRA weights into base model ...")
merged = peft_model.merge_and_unload()
print("    Merge complete ✓")

# ── 6. Save locally ────────────────────────────────────────────────────────────
print(f"\n[6] Saving merged model to {LOCAL_DIR} ...")
os.makedirs(LOCAL_DIR, exist_ok=True)
merged.save_pretrained(LOCAL_DIR, safe_serialization=True)
processor.save_pretrained(LOCAL_DIR)
print("    Saved ✓")

# ── 7. Push to HF via upload_folder ───────────────────────────────────────────
print(f"\n[7] Pushing merged model to {MERGED_REPO} ...")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=MERGED_REPO,
    repo_type="model",
    commit_message="Add merged full model (base + LoRA weights fused)",
    token=TOKEN,
)
print("    Push complete ✓")

# ── 8. Upload README ──────────────────────────────────────────────────────────
readme = f"""---
license: apache-2.0
language:
- or
tags:
- ocr
- odia
- vision
- qwen2.5-vl
- fine-tuned
pipeline_tag: image-text-to-text
base_model: {BASE_MODEL}
---

# OdiaGenAI OCR — Merged Full Model

This is the **fully-merged** version of the Odia OCR fine-tuned model.  
LoRA adapter weights (v2, trained on 145K Odia OCR samples) from [`{ADAPTER}`](https://huggingface.co/{ADAPTER}) have been merged into the base [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) checkpoint, producing a single stand-alone model ready for direct inference via the HF Inference API.

## Usage

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from PIL import Image

model_id = "{MERGED_REPO}"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

image = Image.open("your_odia_image.png").convert("RGB")
messages = [{{
    "role": "user",
    "content": [
        {{"type": "image", "image": image}},
        {{"type": "text",  "text": "Extract all Odia text from this image. Return only the text."}},
    ],
}}]

text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
result = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
```

## Training details
- **Base model**: {BASE_MODEL}
- **Fine-tuning dataset**: [`shantipriya/odia-ocr-merged`](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) — 145K Odia OCR samples
- **Method**: LoRA (r=16, alpha=32) fine-tuning via PEFT
- **LoRA adapter**: [`{ADAPTER}`](https://huggingface.co/{ADAPTER})
- **Organisation**: [OdiaGenAI](https://huggingface.co/OdiaGenAIOCR)
"""

api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=MERGED_REPO,
    repo_type="model",
    commit_message="Add README for merged model",
    token=TOKEN,
)
print("    README uploaded ✓")

print("\n" + "=" * 70)
print("  DONE!")
print(f"  Model: https://huggingface.co/{MERGED_REPO}")
print("=" * 70)
