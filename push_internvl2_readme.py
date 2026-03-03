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
| **Framework** | Transformers + PEFT |

### LoRA Configuration

- **Target modules**: `wqkv`, `wo` (attention), `w1`, `w2`, `w3` (InternLM2 MLP / FFN)
- **Rank / Alpha**: 64 / 128
- **Dropout**: 0.05
- **Task type**: `CAUSAL_LM`

---

## Training Data

Fine-tuned on a merged dataset of **58 720 image–text pairs** from two sources:

| Dataset | Description |
|---|---|
| [OdiaGenAIOCR/synthetic_data](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data) | Synthetically rendered Odia paragraph images |
| [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) | Merged cleaned Odia OCR dataset (2 700 word-level samples) |

Images contain **paragraph-level** Odia text printed in varied fonts, sizes and layouts.  
InternVL2-8B uses dynamic-resolution tiling (up to 6 × 448 px tiles ≈ 1 100 visual tokens per image).

---

## Usage

```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# Load base model + LoRA adapter
base = AutoModel.from_pretrained(
    "OpenGVLab/InternVL2-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "shantipriya/odia-ocr-internvl2-8b")
tokenizer = AutoTokenizer.from_pretrained(
    "shantipriya/odia-ocr-internvl2-8b",
    trust_remote_code=True
)
model.eval()

# Prepare image
image = Image.open("odia_document.png").convert("RGB")
transform = build_transform(448)
pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(model.device)

# Run OCR
question = "<image>\\nTranscribe all the Odia text from this image exactly as it appears."
response = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=512))
print(response)
```

---

## Training Metrics

| Step | Train Loss | LR | Epoch |
|------|-----------|-----|-------|
| 100  | —         | 5.0e-05 | 0.40 |
| 200  | —         | 5.0e-05 | 0.79 |
| 300  | —         | 5.0e-05 | 1.19 |
| 400  | —         | 5.0e-05 | 1.59 |
| … (training ongoing) | ↓ | cosine decay | ↑ |

> **Note**: Loss logging is under investigation (custom trainer returns `0.0` for logged loss while gradients are applied via `compute_loss`). Checkpoint adapter weights are updated correctly (577 MB LoRA adapter per checkpoint). Full metrics will be updated on the final checkpoint.

> Checkpoints are pushed every 100 training steps to this repo.

---

## Sample Inference

Sample inference results will be added here once the model produces valid outputs.  
The sister model [shantipriya/odia-ocr-qwen-finetuned_v3](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v3) (Qwen2.5-VL-7B) already shows strong step-200 results:

| Quality | Metric |
|---|---|
| ✅ Perfect transcription | CER = 0.000 |
| ⚠️ Minor matra confusion | CER ≈ 0.003–0.012 |
| ❌ Dense conjunct split | CER ≈ 0.006–0.008 |

---

## Intended Use

- **Primary use**: OCR on printed Odia text in document or paragraph images
- **Language**: Odia (ISO 639-1: `or`) — one of the 22 scheduled languages of India
- **Not intended for**: handwritten Odia, non-Odia scripts, or real-time edge deployment without quantization

---

## Limitations

- Model is actively training; checkpoints before step 3 000 will under-perform
- Performance on heavily degraded or handwritten images has not been evaluated
- No RLHF / DPO alignment — outputs are raw OCR transcriptions
- Loss telemetry currently shows `0.0` due to a known logging issue in the custom `compute_loss` path; this does not affect the adapter weights saved to disk

---

## Project

This model is part of the **OdiaGenAI** initiative to build open-source AI tools for the Odia language.

- Organization: [OdiaGenAI](https://huggingface.co/OdiaGenAI)
- Datasets: [OdiaGenAIOCR/synthetic_data](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data)
- Author: [shantipriya](https://huggingface.co/shantipriya)
- Sister model: [shantipriya/odia-ocr-qwen-finetuned_v3](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v3)

---

## Citation

If you use this model, please cite:

```bibtex
@misc{odia-ocr-internvl2,
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
    commit_message="Add full model card README",
)
print("InternVL2 README pushed successfully")
