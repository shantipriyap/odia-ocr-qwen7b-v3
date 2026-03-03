from huggingface_hub import HfApi

readme = """\
---
base_model: OpenGVLab/InternVL2-8B
library_name: peft
pipeline_tag: image-text-to-text
language:
- or
license: apache-2.0
tags:
  - odia
  - ocr
  - lora
  - peft
  - internvl2
  - indic-languages
  - document-ocr
  - fine-tuned
datasets:
  - shantipriya/odia-ocr-merged
  - OdiaGenAIOCR/synthetic_data
---

# Odia OCR — InternVL2-8B LoRA Fine-tune

A **LoRA adapter** fine-tuned on top of [OpenGVLab/InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) for **Odia script Optical Character Recognition (OCR)**.  
The model is trained to extract printed and synthetic Odia text from paragraph-level document images.

---

## Model Details

| Field | Value |
|---|---|
| **Base model** | `OpenGVLab/InternVL2-8B` |
| **Adapter type** | LoRA (rank 64, alpha 128) |
| **Task** | Image → Odia text transcription |
| **Script** | Odia (ଓଡ଼ିଆ) |
| **Training steps** | 3 000 (ongoing — checkpoints pushed every 100 steps) |
| **Batch size** | 2 × gradient-accumulation 16 = effective 32 |
| **Hardware** | 1 × NVIDIA A100 80 GB |
| **Framework** | Transformers + PEFT + TRL |
| **Speed** | ~8.6 s/it |

### LoRA Configuration

- **Target modules**: `wqkv`, `wo`, `w1`, `w2`, `w3`
- **Rank / Alpha**: 64 / 128
- **Dropout**: 0.05

### Dynamic Resolution

InternVL2-8B uses **dynamic tiling** — each image is split into up to 6 tiles of 448×448 px, giving up to 1 280 visual tokens per image. This improves OCR accuracy on high-resolution document scans.

---

## Training Data

Fine-tuned on a merged dataset of **8 049 image–text pairs** from two sources:

| Dataset | Samples | Description |
|---|---|---|
| [OdiaGenAIOCR/synthetic_data](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data) | 5 349 | Synthetically rendered Odia paragraph images |
| [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) | 2 700 | Cleaned Odia OCR dataset |

Images contain **paragraph-level** Odia text printed in varied fonts, sizes and layouts.

---

## Usage

```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from PIL import Image
import torch

# Load base model + LoRA adapter
base = AutoModel.from_pretrained(
    "OpenGVLab/InternVL2-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "shantipriya/odia-ocr-internvl2-8b")
tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL2-8B", trust_remote_code=True
)

# Run OCR on an image
image = Image.open("odia_document.png").convert("RGB")

# Build prompt with image context tokens
n_tiles = 1   # adjust based on your tiling
img_tokens = "<IMG_CONTEXT>" * (256 * n_tiles)
prompt = f"<img>{img_tokens}</img>\\nTranscribe all the Odia text from this image exactly as it appears."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Training Metrics

Training loss is decreasing steadily (step 34/3000 at time of publish):

| Step | Notes |
|------|-------|
| 0 | Training started — model loaded (26 GB on A100) |
| 34 | ~1 % through training, ~8.6 s/it |
| … | Checkpoints pushed every 100 steps |

> Full loss + CER table will be updated as training progresses toward step 3 000.

---

## Comparison with Qwen Sibling Model

This model is trained alongside a Qwen2.5-VL-7B sibling on the same data:

| Model | Repo |
|---|---|
| Qwen2.5-VL-7B LoRA | [shantipriya/odia-ocr-qwen-finetuned_v3](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v3) |
| **InternVL2-8B LoRA** | **this repo** |

---

## Intended Use

- **Primary use**: OCR on printed Odia text in document or paragraph images
- **Language**: Odia (ISO 639-1: `or`) — one of the 22 scheduled languages of India
- **Not intended for**: handwritten Odia, non-Odia scripts, or real-time edge deployment without quantization

---

## Limitations

- Model is actively training; early checkpoints will under-perform
- Performance on heavily degraded or handwritten images has not been evaluated
- No RLHF / DPO alignment — outputs are raw OCR transcriptions

---

## Project

This model is part of the **OdiaGenAI** initiative to build open-source AI tools for the Odia language.

- Organization: [OdiaGenAI](https://huggingface.co/OdiaGenAI)
- Datasets: [OdiaGenAIOCR/synthetic_data](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data)
- Author: [shantipriya](https://huggingface.co/shantipriya)

---

## Citation

```bibtex
@misc{odia-ocr-internvl2-8b,
  author    = {Shantipriya Parida},
  title     = {Odia OCR InternVL2-8B LoRA Fine-tune},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/shantipriya/odia-ocr-internvl2-8b}
}
```

---

## License

Apache 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
"""

api = HfApi(token="YOUR_HF_TOKEN_HERE")

with open("/tmp/README_internvl2.md", "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="/tmp/README_internvl2.md",
    path_in_repo="README.md",
    repo_id="shantipriya/odia-ocr-internvl2-8b",
    repo_type="model",
    commit_message="Add model card README",
)
print("README pushed successfully")
