#!/usr/bin/env python3
"""
Build and push the HF README for shantipriya/odia-ocr-qwen-finetuned_v3
with sample images embedded for both in-domain and out-of-domain data.
Run on server: python3 push_readme_v5.py
"""
import json
import textwrap
from huggingface_hub import HfApi

HF_TOKEN   = "YOUR_HF_TOKEN_HERE"
MODEL_REPO = "shantipriya/odia-ocr-qwen-finetuned_v3"
BASE_URL   = f"https://huggingface.co/{MODEL_REPO}/resolve/main/assets/samples"

# ── Load eval results ─────────────────────────────────────────────────────────
with open("/tmp/eval_full_ckpt300_results.json") as f:
    eval_data = json.load(f)
bench_results  = {s["idx"]: s for s in eval_data["benchmark"]}
indomain_known = {r["idx"]: r for r in eval_data["indomain"]}  # idx 0,1,2

try:
    with open("/tmp/infer_new_indom_results.json") as f:
        new_indom = json.load(f)
    # keys are strings in JSON
    new_indom = {int(k): v for k, v in new_indom.items()}
except FileNotFoundError:
    new_indom = {}
    print("WARNING: /tmp/infer_new_indom_results.json not found — CER for idx 10,50 will show as N/A")

all_indom = {**indomain_known, **new_indom}
INDOMAIN_IDX = [0, 1, 2, 10, 50]

# ── Sample selections ─────────────────────────────────────────────────────────
BENCH_SEL = {
    "printed":    {"good": 53,  "mixed": 94,  "bad":  93},
    "scene_text": {"good": 120, "mixed": 140, "bad": 138},
    "handwritten":{"good": 39,  "mixed": 40,  "bad":  50},
    "Digital":    {"good": 5,   "mixed": 4,   "bad":   0},
    "Book":       {"good": 17,  "mixed": 15,  "bad":  19},
    "Newspaper":  {"good": 24,  "mixed": 30,  "bad":  27},
}

def quality(cer):
    if cer is None: return "—"
    if cer < 0.4:   return "Good"
    elif cer < 0.8: return "Mixed"
    else:           return "Bad"

def wrap_text(t, w=55):
    return "<br>".join(textwrap.wrap(str(t)[:200], w))

def img_tag(fname, width=200):
    return f'<img src="{BASE_URL}/{fname}" width="{width}"/>'

# ── Build "Sample Inference" section ─────────────────────────────────────────
def build_samples_section():
    lines = []
    lines.append("## 📸 Sample Inference Results — Checkpoint 300\n")
    lines.append("> Images from **out-of-domain** (`Iftesha/odia-ocr-benchmark`, 6 categories × 3 quality levels)")
    lines.append("> and **in-domain** (`OdiaGenAIOCR/synthetic_data`, 5 paragraph samples).  ")
    lines.append("> Quality labels: ✅ **Good** (CER < 0.4) · ⚠️ **Mixed** (CER 0.4–0.8) · ❌ **Bad** (CER > 0.8)\n")
    lines.append("---\n")

    # ── out-of-domain ──────────────────────────────────────────────────────────
    lines.append("### 🔤 Out-of-Domain Benchmark (`Iftesha/odia-ocr-benchmark`)\n")
    lines.append("Six categories from the benchmark — one good, mixed, and bad sample each.\n")

    EMOJI = {"Good": "✅", "Mixed": "⚠️", "Bad": "❌", "—": ""}

    for cat, sel in BENCH_SEL.items():
        lines.append(f"<details>")
        lines.append(f"<summary><b>{cat}</b> — click to expand</summary>\n")
        lines.append("| Quality | Image | Ground Truth | Predicted | CER |")
        lines.append("|:-------:|:-----:|:------------|:----------|:---:|")
        for q_label, idx in [("Good", sel["good"]), ("Mixed", sel["mixed"]), ("Bad", sel["bad"])]:
            r = bench_results[idx]
            cer_val = r["cer"]
            q = quality(cer_val)
            lines.append(
                f"| {EMOJI.get(q, '')} {q} "
                f"| {img_tag(f'bench_{idx}.png')} "
                f"| {wrap_text(r['gt'])} "
                f"| {wrap_text(r['pred'])} "
                f"| **{cer_val:.3f}** |"
            )
        lines.append("\n</details>\n")

    # ── in-domain ──────────────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("### 📄 In-Domain Samples (`OdiaGenAIOCR/synthetic_data`)\n")
    lines.append("Five synthetic paragraph images — three are from the step-300 evaluation, two are additional samples.\n")
    lines.append("| # | Image | Ground Truth | Predicted | CER | Quality |")
    lines.append("|:-:|:-----:|:------------|:----------|:---:|:-------:|")

    for i, idx in enumerate(INDOMAIN_IDX, 1):
        r = all_indom.get(idx, {})
        cer_val = r.get("cer")
        cer_str = f"**{cer_val:.3f}**" if cer_val is not None else "—"
        q = quality(cer_val)
        emoji = EMOJI.get(q, "")
        lines.append(
            f"| {i} "
            f"| {img_tag(f'indomain_{idx}.png')} "
            f"| {wrap_text(r.get('gt', ''))} "
            f"| {wrap_text(r.get('pred', ''))} "
            f"| {cer_str} "
            f"| {emoji} {q} |"
        )

    lines.append("")
    return "\n".join(lines)

SAMPLES_SECTION = build_samples_section()

# ── Build full README ─────────────────────────────────────────────────────────
readme = f"""\
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
  - Iftesha/odia-ocr-benchmark
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
| **Training steps** | 3 000 (ongoing — latest: step 300/3000) |
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
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Run OCR on an image
image = Image.open("odia_document.png").convert("RGB")

messages = [{{
    "role": "user",
    "content": [
        {{"type": "image", "image": image}},
        {{"type": "text",  "text": "Transcribe all the Odia text from this image exactly as it appears."}}
    ]
}}]

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
| … (training ongoing) | ↓ | ↓ | ↑ | |

> Checkpoints are pushed every 100 training steps.  
> Accuracy is reported as **1 − CER** (character-level).  
> ⚠️ **Eval dataset note**: Steps 0/100/200 used **word-level** images (`shantipriya/odia-ocr-merged`, single words), which yields lower CER. Step 300 used **paragraph-level** images (`OdiaGenAIOCR/synthetic_data`, full paragraphs, ~300 chars) — a much harder task. The lower accuracy at step 300 reflects the harder benchmark, not regression. Full paragraph-level evaluation will be the standard going forward.

---

{SAMPLES_SECTION}

---

## Benchmark Summary (Iftesha/odia-ocr-benchmark — 151 samples)

| Category | Samples | Avg CER | Accuracy |
|----------|--------:|--------:|--------:|
| scene\\_text | 50 | 0.637 | **36.3%** |
| handwritten | 19 | 0.663 | **33.7%** |
| Digital | 10 | 0.705 | 29.5% |
| Book | 11 | 0.906 | 9.4% |
| Newspaper | 11 | 0.941 | 5.9% |
| printed | 50 | 1.287 | — † |
| **Overall** | **151** | **0.902** | **9.9%** |

> † CER > 1 on `printed` indicates hallucination — model generates longer output than the ground truth. This is expected early in training.  
> Best performance on **scene_text** and **handwritten** at this early checkpoint (step 300/3000).

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

```bibtex
@misc{{odia-ocr-qwen-v3,
  author    = {{Shantipriya Parida}},
  title     = {{Odia OCR Qwen2.5-VL LoRA Fine-tune v3}},
  year      = {{2026}},
  publisher = {{HuggingFace}},
  url       = {{https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v3}}
}}
```

---

## License

Apache 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
"""

api = HfApi(token=HF_TOKEN)
with open("/tmp/README_v5.md", "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="/tmp/README_v5.md",
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Update README: add sample images (in-domain + out-of-domain, good/mixed/bad)",
)
print("README v5 pushed successfully")
