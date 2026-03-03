from huggingface_hub import HfApi

readme = """\
---
base_model: Qwen/Qwen2.5-VL-7B-Instruct
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
  - qwen2.5-vl
  - indic-languages
  - document-ocr
  - fine-tuned
datasets:
  - shantipriya/odia-ocr-merged
  - OdiaGenAIOCR/synthetic_data
---

# Odia OCR — Qwen2.5-VL-7B LoRA Fine-tune (v3)

A **LoRA adapter** fine-tuned on top of [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) for **Odia script Optical Character Recognition (OCR)**.  
The model is trained to extract printed and synthetic Odia text from paragraph-level document images.

---

## Model Details

| Field | Value |
|---|---|
| **Base model** | `Qwen/Qwen2.5-VL-7B-Instruct` |
| **Adapter type** | LoRA (rank 16, alpha 32) |
| **Task** | Image → Odia text transcription |
| **Script** | Odia (ଓଡ଼ିଆ) |
| **Training steps** | 3 000 (ongoing — latest: step 900/3000) |
| **Batch size** | 2 × gradient-accumulation 8 = effective 16 |
| **Hardware** | 1 × NVIDIA A100 80 GB |
| **Framework** | Transformers + PEFT + TRL |

### LoRA Configuration

- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Rank / Alpha**: 16 / 32
- **Dropout**: 0.05

---

## Training Data

Fine-tuned on a merged dataset of **58 720 image–text pairs** from two sources:

| Dataset | Description |
|---|---|
| [OdiaGenAIOCR/synthetic_data](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data) | Synthetically rendered Odia paragraph images |
| [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) | Merged cleaned Odia OCR dataset |

Images contain **paragraph-level** Odia text printed in varied fonts, sizes and layouts.

---

## Usage

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# Load base model + LoRA adapter
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "shantipriya/odia-ocr-qwen-finetuned_v3")
processor = AutoProcessor.from_pretrained("shantipriya/odia-ocr-qwen-finetuned_v3")

# Run OCR on an image
image = Image.open("odia_document.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Transcribe all the Odia text from this image exactly as it appears."}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512)

output = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]
print(output)
```

---

## Training Metrics

Training loss drops sharply as the model adapts to Odia OCR:

| Step | Train Loss | CER (5 samples) | Accuracy (1 − CER) | Eval dataset |
|------|-----------|-----------------|-------------------|--------------|
| 0 (baseline) | — | 0.861 | 13.9 % | word-level |
| 25  | 0.797 | — | — | — |
| 50  | 0.673 | — | — | — |
| 75  | 0.447 | — | — | — |
| 100 | 0.295 | 0.603 | 39.7 % | word-level |
| 125 | 0.223 | — | — | — |
| 150 | 0.185 | — | — | — |
| 175 | 0.159 | — | — | — |
| 200 | 0.138 | 0.504 | **49.6 %** | word-level |
| 225 | 0.134 | — | — | — |
| 250 | 0.120 | — | — | — |
| 275 | 0.119 | — | — | — |
| 300 | 0.110 | 0.763 | **23.7 %** | paragraph-level† |
| **400** | **0.072** | — | — | — |
| 500 | 0.070 | — | — | — |
| 600 | 0.054 | — | — | — |
| 700 | 0.047 | — | — | — |
| 800 | 0.043 | — | — | — |
| **900** | **0.034** | — | — | — |
| … (training ongoing) | ↓ | ↓ | ↑ | |

> Checkpoints are pushed every 100 training steps.  
> Accuracy is reported as **1 − CER** (character-level).  
> ⚠️ **Eval dataset note**: Steps 0/100/200 used **word-level** images (`shantipriya/odia-ocr-merged`, single words), which yields lower CER. Step 300 used **paragraph-level** images (`OdiaGenAIOCR/synthetic_data`, full paragraphs, ~300 chars) — a much harder task. The lower accuracy at step 300 reflects the harder benchmark, not regression. Full paragraph-level evaluation will be the standard going forward.

---

## Sample Inference — Checkpoint Updates

> Latest pushed checkpoint: **checkpoint-900** (train loss **0.034**). Training is at step 959/3000 (~32%). Checkpoints 700, 800, 900 have been pushed. Full benchmark evaluation will be updated upon training completion.

---

## Step 300 Checkpoint Results

### In-Domain Long Paragraphs (OdiaGenAIOCR/synthetic_data)

Evaluated on 3 long paragraph samples (>200 chars) at checkpoint-300:

| # | CER | Ground Truth (truncated) | Model Output (truncated) |
|---|-----|--------------------------|--------------------------|
| 1 | 0.815 | ଗୋପକୁ ଗଲେ କୃଷ୍ଣ ବରଗଡ଼ (ଦୀପକ ଶର୍ମା): ମଥୁରାନଗରୀ ସାଜିଥିବା ବରଗଡ଼ରେ ଚାଲିଛି ପ୍ରବଳ ପ୍ରତାପି ମହାରାଜ… | ଯୋଦ୍ଧାକୁ ଘରେ ଦୃଢ଼ ବଜରଡ଼ ଡାଏର ଶର୍ମା ମନ୍ତ୍ରଣାଳୟର ଗାଢିଥିବା ବଜରଡ଼ରେ ତାଲିଛି। ପୂର୍ବକ ପ୍ରତାପି ମହାରାଜ… |
| 2 | 0.706 | ଖେଳୁଥିବା ସମୟରେ ଆକାଶରୁ ଖସିଲା ନିଆଁ, ଚାଲିଗଲା ନାବାଳକର ଜୀବନ କରଞ୍ଜିଆ(ଓଡ଼ିଶା ରିପୋର୍ଟର): ଆକାଶରୁ… | ଶୋକୁଥିବା ସମୟରେ ଆକାଶରୁ ଖସିଲା ନିଆଁ, ଚାଲିଗଲା ନାବାଳକର ଜୀବନ କରିଲାଣାଓଡ଼ିଶା ରିପୋର୍ଟର ଆକାଶରୁ… |
| 3 | 0.790 | ନମସ୍କାର ବନ୍ଧୁଗଣ ତେବେ ଧନ ତ୍ରୟୋଦଶୀ ଦିନ ପ୍ରାୟତଃ ଧନତେରସ ପୂଜା କରାଯାଏ । ହିନ୍ଦୁ ଧର୍ମରେ ଧନ୍ତେରସ ସୁଖ… | ନମ୍ବୋର ବନ୍ଧୁଗଣ ତେବେ ଧନ ତ୍ରୟୋଦଶୀ ଦିନ ପ୍ରାୟତଃ ଧନରେଷ ସୂଚୀ କରାଯାଏ । କିନ୍ତୁ ଧର୍ମରେ ଧନରେଷ ସୁଖ… |
| **Avg** | **0.770** | | |

> ⚠️ These are **paragraph-level newspaper/synthetic images** — a significantly harder task than word-level OCR. Model is at step 300/3000 (~10% through training).

---

### Out-of-Domain Benchmark (Iftesha/odia-ocr-benchmark — 151 samples)

Evaluated on all 151 samples across 6 categories at checkpoint-300:

| Category | Samples | Avg CER | Accuracy |
|----------|--------:|--------:|--------:|
| scene\_text | 50 | 0.637 | **36.3%** |
| handwritten | 19 | 0.663 | **33.7%** |
| Digital | 10 | 0.705 | 29.5% |
| Book | 11 | 0.906 | 9.4% |
| Newspaper | 11 | 0.941 | 5.9% |
| printed | 50 | 1.287 | — † |
| **Overall** | **151** | **0.902** | **9.9%** |

> † CER > 1 on `printed` indicates hallucination — model generates longer output than the ground truth. This is expected early in training and will improve as training progresses.  
> Best performance on **scene_text** and **handwritten** categories at this early checkpoint.  
> Benchmark: [Iftesha/odia-ocr-benchmark](https://huggingface.co/datasets/Iftesha/odia-ocr-benchmark)

---

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

---

## Project

This model is part of the **OdiaGenAI** initiative to build open-source AI tools for the Odia language.

- Organization: [OdiaGenAI](https://huggingface.co/OdiaGenAI)
- Datasets: [OdiaGenAIOCR/synthetic_data](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data)
- Author: [shantipriya](https://huggingface.co/shantipriya)

---

## Citation

If you use this model, please cite:

```bibtex
@misc{odia-ocr-qwen-v3,
  author    = {Shantipriya Parida},
  title     = {Odia OCR Qwen2.5-VL LoRA Fine-tune v3},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v3}
}
```

---

## License

Apache 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
"""

api = HfApi(token="YOUR_HF_TOKEN_HERE")

with open("/tmp/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="/tmp/README.md",
    path_in_repo="README.md",
    repo_id="shantipriya/odia-ocr-qwen-finetuned_v3",
    repo_type="model",
    commit_message="Update README: add checkpoints 500-900 with loss values",
)
print("README pushed successfully")
