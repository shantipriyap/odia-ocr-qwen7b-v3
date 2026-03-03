#!/usr/bin/env python3
"""Push updated README to OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged"""
from huggingface_hub import HfApi

TOKEN = "os.getenv("HF_TOKEN", "")"
REPO  = "OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged"

readme = """\
---
license: apache-2.0
language:
- or
tags:
- ocr
- odia
- vision
- qwen2.5-vl
- fine-tuned
- merged
- image-to-text
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen2.5-VL-3B-Instruct
datasets:
- shantipriya/odia-ocr-merged
model-index:
- name: OdiaGenAI OCR Qwen2.5-VL Merged
  results: []
---

# OdiaGenAI OCR — Qwen2.5-VL-3B (Merged Full Model)

A **fully stand-alone** Odia OCR model, obtained by merging LoRA fine-tuning weights into the base [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) checkpoint.  
No adapter or PEFT library is required at inference time — just load and run.

> **Organisation:** [OdiaGenAI](https://huggingface.co/OdiaGenAIOCR)  
> **Task:** Optical Character Recognition (OCR) for Odia (ଓଡ଼ିଆ) script  
> **Model size:** ~3B parameters (7.5 GB in fp16)

---

## Model Details

| Property | Value |
|---|---|
| **Base model** | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Fine-tuning method** | LoRA (r=16, alpha=32) via PEFT |
| **Merge method** | `PeftModel.merge_and_unload()` |
| **Training data** | [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) |
| **Training samples** | 145,000 Odia OCR image-text pairs |
| **LoRA adapter (v2)** | [shantipriya/odia-ocr-qwen-finetuned_v2](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v2) |
| **Language** | Odia (ଓଡ଼ିଆ) — `or` |
| **Precision** | float16 |
| **GPU required** | Recommended (works on ≥16 GB VRAM) |

---

## Quick Start

### Installation

```bash
pip install transformers torch pillow accelerate
```

### Inference

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from PIL import Image

model_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# Run OCR on an image
image = Image.open("odia_text_image.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Extract all Odia text from this image. Return only the text."},
    ],
}]

text_input = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text_input], images=[image], return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

# Decode only the generated portion
input_len = inputs["input_ids"].shape[1]
result = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
print(result)
```

### Batch inference with multiple images

```python
images = [Image.open(p).convert("RGB") for p in image_paths]

all_inputs = []
for img in images:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": "Extract all Odia text from this image. Return only the text."},
        ],
    }]
    all_inputs.append(
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    )

inputs = processor(text=all_inputs, images=images, padding=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

results = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

---

## Training Details

The model was trained on **145,000 Odia OCR samples** from the merged dataset [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged), combining:

- Word-level printed Odia text images
- Line-level Odia text samples
- Sources: historical manuscripts, newspapers, books, and digital documents

### LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Training Run

| Parameter | Value |
|---|---|
| Max steps | 6,400 |
| Learning rate | 1e-4 (cosine decay) |
| Warmup steps | 100 |
| Batch size | 4 (per device) |
| Gradient accumulation | 4 steps |
| Optimizer | AdamW |
| Precision | bf16 |
| Hardware | NVIDIA H100 80GB |

---

## Sample Outputs

**Input:** Odia handwritten/printed text image  
**Prompt:** `"Extract all Odia text from this image. Return only the text."`

| Ground Truth | Model Output |
|---|---|
| `ସୂଚିପତ୍ର` | `ସୃଗପତ୍ରା` |
| `ଅବସର ବାସରେ` | `ଅବସର ବାସରେ` |
| `ଶ୍ରୀ ଫକୀରମୋହନ ସେନାପତି` | `ଶ୍ରୀ ଫକୀରମୋହନ ସେନାପତି` |

> **Note:** The model was fine-tuned primarily on word- and line-level images. For full-page OCR, consider splitting the image into horizontal strips (~400px height) before inference.

---

## Related Resources

| Resource | Link |
|---|---|
| LoRA Adapter (v2) | [shantipriya/odia-ocr-qwen-finetuned_v2](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v2) |
| Merged Model (this) | [OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged) |
| v1 Merged Model | [OdiaGenAIOCR/odia-ocr-qwen-finetuned](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned) |
| Training Dataset | [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) |
| Paragraph Dataset | [OdiaGenAIOCR/Odia-lipi-ocr-data](https://huggingface.co/datasets/OdiaGenAIOCR/Odia-lipi-ocr-data) |
| Base Model | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |

---

## Citation

If you use this model, please cite:

```bibtex
@misc{odiagen-ocr-2025,
  title        = {OdiaGenAI OCR: Fine-tuned Qwen2.5-VL for Odia Script Recognition},
  author       = {OdiaGenAI},
  year         = {2025},
  howpublished = {\\url{https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned-merged}},
  note         = {Merged full model (base + LoRA), trained on 145K Odia OCR samples}
}
```

---

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).  
See also the license of the base model: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
"""

api = HfApi(token=TOKEN)
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
    commit_message="Update README: add full model card with usage, training details, sample outputs",
    token=TOKEN,
)
print(f"README updated: https://huggingface.co/{REPO}")
