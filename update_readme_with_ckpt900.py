"""
Run this after /tmp/eval_full_ckpt900_results.json is produced on the server.
1. SSHes to server and downloads the JSON
2. Parses per-category CER / accuracy
3. Rewrites push_readme.py with the new results
4. Pushes new README to HF
"""

import subprocess, json, sys

SERVER = "root@135.181.63.224"
REMOTE_JSON = "/tmp/eval_full_ckpt1700_results.json"

# ── 1. Download results ──────────────────────────────────────────────────────
print("Downloading eval results from server …")
proc = subprocess.run(
    ["ssh", SERVER, f"cat {REMOTE_JSON}"],
    capture_output=True, text=True, timeout=30
)
if proc.returncode != 0:
    print("ERROR:", proc.stderr)
    sys.exit(1)

data = json.loads(proc.stdout)
samples = data.get("benchmark", data.get("samples", []))
print("Downloaded OK,", len(samples), "benchmark samples")
by_cat: dict[str, list] = {}
for s in samples:
    cat = s.get("category", "unknown")
    by_cat.setdefault(cat, []).append(s.get("cer", 1.0))

# Summary
print("\n=== Per-category CER (checkpoint-1700) ===")
cat_rows = []
for cat in ["scene_text", "handwritten", "Digital", "Book", "Newspaper", "printed"]:
    if cat in by_cat:
        vals = by_cat[cat]
        avg_cer = sum(vals) / len(vals)
        acc = max(0.0, 1.0 - avg_cer) * 100
        cer_str = f"{avg_cer:.3f}"
        acc_str = f"**{acc:.1f}%**" if acc >= 30 else f"{acc:.1f}%"
        cat_rows.append((cat, len(vals), cer_str, acc_str))
        print(f"  {cat:15s} n={len(vals):3d}  CER={avg_cer:.3f}  Acc={acc:.1f}%")

overall_cer = data.get("benchmark_overall_cer",
              data.get("overall_cer",
              sum(s.get("cer",1) for s in samples)/max(len(samples),1)))
overall_acc = max(0.0, 1.0 - overall_cer) * 100
print(f"\n  {'Overall':15s} n={len(samples):3d}  CER={overall_cer:.3f}  Acc={overall_acc:.1f}%")

# ── 2b. Download sample manifest (images already uploaded to HF) ─────────────
print("\nDownloading sample manifest …")
proc2 = subprocess.run(
    ["ssh", SERVER, "cat /tmp/bench_manifest_ckpt1700.json"],
    capture_output=True, text=True, timeout=30
)
manifest = json.loads(proc2.stdout) if proc2.returncode == 0 else []
print(f"Manifest: {len(manifest)} entries")

# ── 2c. Download ckpt-1300 manifest (best checkpoint samples) ─────────────────
print("\nDownloading ckpt-1300 best-checkpoint manifest …")
proc3 = subprocess.run(
    ["ssh", SERVER, "cat /tmp/bench_manifest_ckpt1300.json"],
    capture_output=True, text=True, timeout=30
)
manifest_1300 = json.loads(proc3.stdout) if proc3.returncode == 0 else []
print(f"Ckpt-1300 manifest: {len(manifest_1300)} entries")

# ── 3. Build the updated README ───────────────────────────────────────────────
cat_table_rows = ""
for cat, n, cer, acc in cat_rows:
    cat_table_rows += f"| {cat} | {n} | {cer} | {acc} |\n"
cat_table_rows += f"| **Overall** | **{len(samples)}** | **{overall_cer:.3f}** | **{overall_acc:.1f}%** |\n"

# Build per-category image tables (5 samples each)
QUALITY_ICON = {"good": "✅", "mixed": "🔶", "bad": "❌"}
CAT_ORDER = ["scene_text", "handwritten", "Digital", "Book", "Newspaper", "printed"]

def escape(text, max_len=90):
    return text[:max_len].replace("|", "&#124;").replace("\n", " ").strip()

def build_cat_section(cat, entries):
    rows = ""
    for e in entries:
        cer_disp = min(float(e["cer"]), 1.0)
        icon = QUALITY_ICON.get(e["quality"], "")
        img_md = f'<img src="{e["hf_url"]}" width="180"/>'
        gt   = escape(e["gt"])
        pred = escape(e["pred"])
        rows += f"| {icon} | {cer_disp:.2f} | {img_md} | {gt} | {pred} |\n"
    return f"""
#### {cat.replace("_", " ").title()}

| Quality | CER | Image | Ground Truth | Prediction |
|---------|-----|-------|--------------|------------|
{rows}"""

# Group manifest by category
from collections import defaultdict
man_by_cat = defaultdict(list)
for e in manifest:
    man_by_cat[e["category"]].append(e)

per_cat_sections = ""
for cat in CAT_ORDER:
    if cat in man_by_cat:
        per_cat_sections += build_cat_section(cat, man_by_cat[cat])

# Build ckpt-1300 best checkpoint samples
man1300_by_cat = defaultdict(list)
for e in manifest_1300:
    man1300_by_cat[e["category"]].append(e)

best_cat_sections = ""
for cat in CAT_ORDER:
    if cat in man1300_by_cat:
        best_cat_sections += build_cat_section(cat, man1300_by_cat[cat])

sample_section = f"""
### Sample Inferences — Best Checkpoint 1300 (CER=0.655, Acc=34.5%) — 5 per category

Checkpoint-1300 is the **best checkpoint overall** — lowest CER and highest accuracy across all categories.
Each row shows the original image, ground truth text, and model prediction.
Quality icons: ✅ Good (CER < 0.15) · 🔶 Mixed (CER 0.15–0.65) · ❌ Bad (CER > 0.65)

{best_cat_sections}
> Evaluated on [Iftesha/odia-ocr-benchmark](https://huggingface.co/datasets/Iftesha/odia-ocr-benchmark) — **out-of-domain** from training data.  
> Best performance: **handwritten** and **scene_text** categories.  
> ⭐ Use `checkpoint-1300` for best inference results.

### Sample Inferences — Latest Checkpoint 1700 (CER=0.912, Acc=8.8%) — 5 per category

Latest evaluated checkpoint (step 1700/3000). Note: model is in overfitting phase — ckpt-1300 gives better results.

{per_cat_sections}
> ⚠️ Model is overfitting after step 1300 — training loss continues to drop but benchmark accuracy is degrading.
"""

from huggingface_hub import HfApi

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
| **Training steps** | 3 000 (completed — best checkpoint: step 1300/3000) |
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
| 400 | 0.072 | — | — | — |
| 500 | 0.070 | — | — | — |
| 600 | 0.054 | — | — | — |
| 700 | 0.047 | — | — | — |
| 800 | 0.043 | — | — | — |
| **900** | **0.034** | **0.804** | **19.6%** | paragraph-level |
| **1000** | **~0.030** | **0.863** | **13.7%** | paragraph-level |
| **1100** | — | — | — | — |
| **1200** | — | — | — | — |
| **1300** | — | **0.655** | **34.5%** | paragraph-level |
| **1400** | **0.015** | **0.690** | **31.0%** | paragraph-level |
| **1500** | **0.012** | **0.690** | **31.0%** | paragraph-level |
| **1600** | **0.010** | **0.758** | **24.2%** | paragraph-level |
| **1700** | **~0.009** | **{overall_cer:.3f}** | **{overall_acc:.1f}%** | paragraph-level |
| **1800** | **0.0085** | pending | pending | paragraph-level |

> ⚠️ **Overfitting note**: Best checkpoint is **1300** (CER=0.655, Acc=34.5%). Performance degrades after step 1300 despite training loss continuing to drop.

### Benchmark CER & Accuracy vs Checkpoint

![Benchmark CER and Accuracy vs Checkpoint](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v3/resolve/main/assets/benchmark_cer_accuracy.png)

> Checkpoints are pushed every 100 training steps.  
> Accuracy is reported as **1 − CER** (character-level).  
> ⚠️ **Eval dataset note**: Steps 0/100/200 used **word-level** images (`shantipriya/odia-ocr-merged`, single words), which yields lower CER. Step 300 used **paragraph-level** images (`OdiaGenAIOCR/synthetic_data`, full paragraphs, ~300 chars) — a much harder task. The lower accuracy at step 300 reflects the harder benchmark, not regression. Full paragraph-level evaluation will be the standard going forward.

---

## ⭐ Best Checkpoint: 1300 (CER=0.655, Acc=34.5%)

Checkpoint-1300 is the best performing checkpoint. Use this for inference:  
`shantipriya/odia-ocr-qwen-finetuned_v3` — load with `revision="checkpoint-1300"`

## Checkpoint-1700 Benchmark Results (151 samples — Iftesha/odia-ocr-benchmark)

Latest eval at **checkpoint-1700** (note: overfitting phase — ckpt-1300 remains best):

| Category | Samples | Avg CER | Accuracy (1−CER) |
|----------|--------:|--------:|----------------:|
{cat_table_rows}
> Benchmark: [Iftesha/odia-ocr-benchmark](https://huggingface.co/datasets/Iftesha/odia-ocr-benchmark)  
> Checkpoint-1700 results (CER={overall_cer:.3f}). History: ckpt-1600 CER=0.758, ckpt-1500 CER=0.690, **ckpt-1300 CER=0.655 (best)**, ckpt-900 CER=0.804.  
> ⭐ **Recommended checkpoint for inference: ckpt-1300** (34.5% accuracy).
{sample_section}

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

### Out-of-Domain Benchmark — Checkpoint 300 (Iftesha/odia-ocr-benchmark — 151 samples)

| Category | Samples | Avg CER | Accuracy |
|----------|--------:|--------:|--------:|
| scene\\_text | 50 | 0.637 | **36.3%** |
| handwritten | 19 | 0.663 | **33.7%** |
| Digital | 10 | 0.705 | 29.5% |
| Book | 11 | 0.906 | 9.4% |
| Newspaper | 11 | 0.941 | 5.9% |
| printed | 50 | 1.287 | — † |
| **Overall** | **151** | **0.902** | **9.9%** |

> † CER > 1 on `printed` indicates hallucination — model generates longer output than the ground truth.

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

# ── 4. Push to HF ─────────────────────────────────────────────────────────────
import os
_token = os.environ.get("HF_TOKEN") or "YOUR_HF_TOKEN_HERE"
api = HfApi(token=_token)

with open("/tmp/README_ckpt1700.md", "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="/tmp/README_ckpt1700.md",
    path_in_repo="README.md",
    repo_id="shantipriya/odia-ocr-qwen-finetuned_v3",
    repo_type="model",
    commit_message=f"Update README: ckpt-1700 results (CER={overall_cer:.3f}), best=ckpt-1300 (CER=0.655, Acc=34.5%), samples from best checkpoint",
)
print(f"\nREADME pushed — CER={overall_cer:.3f}, Acc={overall_acc:.1f}%")
