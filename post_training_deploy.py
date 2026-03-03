#!/usr/bin/env python3
"""
Post-training deployment script for Odia OCR.
Run on the H100 server after training completes.
  - Verifies the model was pushed to HF
  - Uploads a rich model card (README.md)
  - Creates and pushes an HF Spaces app.py for Gradio inference
  - Prints a final verification report
"""

import os, sys, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
MODEL_REPO     = "shantipriya/odia-ocr-qwen-finetuned"
BASE_MODEL     = "Qwen/Qwen2.5-VL-3B-Instruct"
SPACE_REPO     = "shantipriya/odia-ocr-demo"   # Gradio Space

# ──────────────────────────────────────────────────────────────────────────────
# 1. Verify model files on Hub
# ──────────────────────────────────────────────────────────────────────────────
def verify_model():
    from huggingface_hub import list_repo_files, hf_hub_download, HfApi
    api = HfApi(token=HF_TOKEN)
    log.info(f"Checking model repo: {MODEL_REPO}")
    try:
        files = list(list_repo_files(MODEL_REPO, repo_type="model", token=HF_TOKEN))
        log.info(f"  {len(files)} files found:")
        for f in sorted(files)[:20]:
            log.info(f"    {f}")
        return True, files
    except Exception as e:
        log.error(f"  Model not accessible: {e}")
        return False, []


# ──────────────────────────────────────────────────────────────────────────────
# 2. Upload rich model card
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CARD = """\
---
language:
- or
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
datasets:
- shantipriya/odia-ocr-merged
tags:
- ocr
- odia
- vision-language
- peft
- lora
- fine-tuned
pipeline_tag: image-text-to-text
---

# Odia OCR – Qwen2.5-VL Fine-tuned

Fine-tuned vision-language model for **Odia (Odisha script) OCR** based on
[Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

## Training details
| | |
|---|---|
| **Base model** | Qwen/Qwen2.5-VL-3B-Instruct |
| **Dataset** | shantipriya/odia-ocr-merged (~73 K images) |
| **Method** | LoRA (r=128, α=256, 7 modules) |
| **Epochs** | 3 |
| **Batch size** | 16 (effective, bf16) |
| **GPU** | NVIDIA H100 80 GB |
| **Framework** | Transformers + PEFT + TRL |

## Quick inference

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

base = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter = "shantipriya/odia-ocr-qwen-finetuned"

processor = AutoProcessor.from_pretrained(base)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

image = Image.open("odia_text.png").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text": "Extract all Odia text from this image. Return only the text."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Try it
👉 [Live demo on HF Spaces](https://huggingface.co/spaces/shantipriya/odia-ocr-demo)

## License
Apache 2.0
"""

def upload_model_card():
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    log.info("Uploading model card (README.md) …")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_CARD.encode(),
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message="Add model card",
        )
        log.info("  Model card uploaded.")
        return True
    except Exception as e:
        log.error(f"  Failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 3. Create Gradio Spaces app for deployment
# ──────────────────────────────────────────────────────────────────────────────
GRADIO_APP = '''\
import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image

BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER     = "shantipriya/odia-ocr-qwen-finetuned"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("Loading model …")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE, device_map="auto" if DEVICE == "cuda" else None
)
model = PeftModel.from_pretrained(_model, ADAPTER)
model.eval()
print("Model ready.")


def run_ocr(image: Image.Image) -> str:
    if image is None:
        return "Please upload an image."
    image = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Extract all Odia text from this image. Return only the Odia text."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
    result = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


with gr.Blocks(title="Odia OCR") as demo:
    gr.Markdown("# 🔤 Odia OCR\\nExtract Odia (ଓଡ଼ିଆ) text from images using a fine-tuned Qwen2.5-VL model.")
    with gr.Row():
        img_input  = gr.Image(type="pil", label="Upload image")
        txt_output = gr.Textbox(label="Extracted Odia text", lines=10)
    btn = gr.Button("Extract Text", variant="primary")
    btn.click(run_ocr, inputs=img_input, outputs=txt_output)
    gr.Examples(
        examples=[],
        inputs=img_input,
    )

demo.launch()
'''

SPACES_README = """\
---
title: Odia OCR
emoji: 🔤
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - shantipriya/odia-ocr-qwen-finetuned
---

# Odia OCR Demo

Extract Odia (ଓଡ଼ିଆ) text from images using [shantipriya/odia-ocr-qwen-finetuned](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned).
"""

SPACES_REQUIREMENTS = """\
gradio>=4.44.0
torch>=2.1.0
transformers>=4.45.0
peft>=0.13.0
accelerate>=0.26.0
qwen-vl-utils
pillow
"""

def create_gradio_space():
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    log.info(f"Creating/updating Gradio Space: {SPACE_REPO}")
    try:
        api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="gradio",
                        exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=GRADIO_APP.encode(), path_in_repo="app.py",
                        repo_id=SPACE_REPO, repo_type="space",
                        commit_message="Add OCR app")
        api.upload_file(path_or_fileobj=SPACES_README.encode(), path_in_repo="README.md",
                        repo_id=SPACE_REPO, repo_type="space",
                        commit_message="Add Space README")
        api.upload_file(path_or_fileobj=SPACES_REQUIREMENTS.encode(), path_in_repo="requirements.txt",
                        repo_id=SPACE_REPO, repo_type="space",
                        commit_message="Add requirements")
        log.info(f"  Space created: https://huggingface.co/spaces/{SPACE_REPO}")
        return True
    except Exception as e:
        log.error(f"  Space creation failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. Wait for training then run all post-training tasks
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILE    = "/root/odia_ocr/training_improved.log"
CKPT_DIR    = "/root/odia_ocr/checkpoints_improved"
TOTAL_STEPS = 12387

def get_current_step():
    """Parse latest step number from the tqdm progress line."""
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            import re
            # match e.g.  1%|  | 91/12387
            m = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    except:
        pass
    return 0, TOTAL_STEPS

def training_finished():
    """True if training completed (final step reached or 'TRAINING COMPLETE' in log)."""
    try:
        with open(LOG_FILE, "r") as f:
            content = f.read()
        if "TRAINING COMPLETE" in content or "Training completed" in content:
            return True
        # check if trainer printed the final training summary table
        if "train_runtime" in content:
            return True
        step, total = get_current_step()
        return step > 0 and step >= total
    except:
        return False

def monitor_loop():
    log.info("=" * 60)
    log.info("POST-TRAINING WATCHER STARTED")
    log.info("Polling training log every 5 minutes …")
    log.info("=" * 60)
    poll_interval = 300  # 5 minutes
    while True:
        step, total = get_current_step()
        pct = 100.0 * step / total if total else 0
        log.info(f"Progress: {step}/{total} ({pct:.1f}%) …")
        if training_finished():
            log.info("✅ Training detected as COMPLETE!")
            break
        time.sleep(poll_interval)

def main():
    if not HF_TOKEN:
        log.error("HF_TOKEN env var not set – aborting.")
        sys.exit(1)

    # Wait for training to finish
    monitor_loop()

    log.info("=" * 60)
    log.info("STARTING POST-TRAINING DEPLOYMENT")
    log.info("=" * 60)

    # Verify model on Hub
    ok, files = verify_model()
    if not ok:
        log.error("Model not found on Hub — training may not have pushed. Check logs.")
        sys.exit(1)

    # Upload model card
    upload_model_card()

    # Create Gradio Space for demo
    create_gradio_space()

    log.info("=" * 60)
    log.info("DEPLOYMENT COMPLETE")
    log.info(f"  Model : https://huggingface.co/{MODEL_REPO}")
    log.info(f"  Demo  : https://huggingface.co/spaces/{SPACE_REPO}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
