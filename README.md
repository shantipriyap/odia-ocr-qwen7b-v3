---
language:
- or
license: apache-2.0
tags:
- ocr
- odia
- qwen
- fine-tuned
- lora
- vision-language
base_model: Qwen/Qwen2.5-VL-7B-Instruct
datasets:
- OdiaGenAIOCR/synthetic_data
- shantipriya/odia-ocr-merged
---

# Odia OCR — Qwen2.5-VL-7B Phase 3

Fine-tuning [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) for **Odia script OCR** — paragraph-level text recognition from document images.

**GitHub:** [shantipriyap/odia-ocr-qwen7b-v3](https://github.com/shantipriyap/odia-ocr-qwen7b-v3)  
**Fine-tuned model:** [shantipriya/odia-ocr-qwen7b-phase3](https://huggingface.co/shantipriya/odia-ocr-qwen7b-phase3)

---

## Training Setup

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Method | LoRA (r=64, alpha=128) |
| Hardware | 2× NVIDIA A100-SXM4-80GB |
| GPU 0 | Training (HF Trainer, single-GPU) |
| GPU 1 | Eval + HF Hub push every 100 steps |
| Precision | bfloat16 + Flash Attention 2 |
| Effective batch | 32 (batch=2, grad_accum=16) |
| Steps | 3000 |
| Learning rate | 5e-5, warmup=100 |
| Max seq len | 4096 |

## Dataset

- [`OdiaGenAIOCR/synthetic_data`](https://huggingface.co/datasets/OdiaGenAIOCR/synthetic_data) — paragraph-level synthetic Odia OCR
- [`shantipriya/odia-ocr-merged`](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) — 2700 word/line-level samples

## Repository Structure

```
phase3_paragraph/
├── train_2gpu.py          # Main 2-GPU training script (GPU0=train, GPU1=eval+push)
├── qwen_phase3_paragraph_train.py  # Single-GPU training variant
├── eval_qwen_checkpoint.py # Checkpoint evaluation
├── inference.py           # Inference CLI
├── monitor.py             # Real-time training monitor
├── launch.sh              # SLURM/multi-GPU launcher
└── setup.sh               # Server-side environment setup

train_v3.py                # Standalone training script (alternative)
qwen2vl_odia_train.py      # Qwen2VL train wrapper
qwen2vl_odia_eval.py       # Qwen2VL eval wrapper
eval.py                    # Standalone evaluation script
evaluate_lora_ocr.py       # LoRA checkpoint evaluator (CER + accuracy)
benchmark_ocr_models.py    # Multi-model benchmark suite (151 samples)
inference.py               # Inference helper
inference_with_postprocessing.py  # Inference + cleanup pipeline
update_readme_with_ckpt900.py     # Auto-update HF README with checkpoint results
requirements.txt           # Python dependencies
Dockerfile                 # Container build
setup_venv.sh              # Local venv setup
```

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/shantipriyap/odia-ocr-qwen7b-v3
cd odia-ocr-qwen7b-v3

# Option A — venv
bash setup_venv.sh

# Option B — manual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies:**
```
torch>=2.1.0
transformers>=4.45.0
peft>=0.12.0
trl>=0.11.0
accelerate>=0.34.0
datasets>=2.20.0
Pillow
huggingface_hub
jiwer        # for CER metric
```

**Hardware requirements:**
- Training: 1–2× A100/H100 80 GB recommended (LoRA r=64 + 7B model)
- Inference: 1× GPU with ≥ 24 GB VRAM (bfloat16), or CPU with quantization
- Minimum for eval: 1× A100 40 GB

### 2. HuggingFace Authentication

```bash
huggingface-cli login
# or set env variable:
export HF_TOKEN=your_token_here
```

---

## Training

### Single-node, 2-GPU (recommended)

Uses GPU 0 for training and GPU 1 for async eval + HF Hub push every 100 steps:

```bash
cd phase3_paragraph
python train_2gpu.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name shantipriya/odia-ocr-merged \
    --output_dir ./output_2gpu \
    --max_steps 3000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --push_to_hub True \
    --hub_model_id shantipriya/odia-ocr-qwen-finetuned_v3
```

### Single-GPU

```bash
python qwen2vl_odia_train.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir ./output \
    --max_steps 3000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --fp16 False \
    --bf16 True
```

### Key Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank `r` | 64 | Higher = more capacity |
| LoRA alpha | 128 | Effective scale = alpha/r = 2.0 |
| LoRA target modules | `q_proj, v_proj, k_proj, o_proj` | Attention layers only |
| Dropout | 0.05 | LoRA dropout |
| Max seq length | 4096 | Image + text tokens |
| LR scheduler | cosine | With 100-step warmup |
| Optimizer | adamw_torch_fused | |

### Monitoring Training

```bash
# Real-time loss/step monitor
python phase3_paragraph/monitor.py

# Or tail the training log
tail -f /tmp/train.log
```

---

## Evaluation

### Evaluate a specific checkpoint

Runs the 151-sample `Iftesha/odia-ocr-benchmark` suite (6 categories: handwritten, scene_text, digital, book, newspaper, printed):

```bash
python eval.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --checkpoint shantipriya/odia-ocr-qwen-finetuned_v3 \
    --revision checkpoint-1500 \
    --output_json /tmp/eval_results.json \
    --gpu_id 0
```

### Evaluate a local checkpoint

```bash
python evaluate_lora_ocr.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --lora_path ./output_2gpu/checkpoint-1500 \
    --output_json /tmp/eval_ckpt1500.json
```

### Metrics

- **CER** (Character Error Rate) — lower is better; computed via `jiwer`
- **Accuracy** — exact match at sentence level (%)
- Results are broken down per image category

### Reading Results

```python
import json
d = json.load(open("/tmp/eval_results.json"))
print(f"Overall CER: {d['benchmark_overall_cer']:.3f}")
print(f"Overall Accuracy: {d['benchmark_overall_acc']:.1f}%")
for cat, v in d["benchmark_summary"].items():
    print(f"  {cat}: CER={v['avg_cer']:.3f}  Acc={v['accuracy']:.1f}%  n={v['count']}")
```

### Multi-model Benchmark

Compare multiple models/checkpoints on the same benchmark:

```bash
python benchmark_ocr_models.py
```

---

## Checkpoint Progress

| Checkpoint | CER ↓ | Accuracy ↑ | Train Loss |
|---|---|---|---|
| 300 | 0.902 | 9.9% | 0.110 |
| 900 | 0.804 | 19.6% | 0.034 |
| 1300 | **0.655** | **34.5%** | — |
| 1500 | 0.690 | 31.0% | 0.012 |

Per-category results at checkpoint-1500:

| Category | CER | Accuracy | n |
|---|---|---|---|
| handwritten | 0.276 | 72.4% | 19 |
| scene_text | 0.473 | 52.7% | 50 |
| digital | 0.703 | 29.7% | 10 |
| book | 0.898 | 10.2% | 11 |
| newspaper | 0.938 | 6.2% | 11 |
| printed | 0.960 | 4.0% | 50 |

---

## Inference

### CLI

```bash
python phase3_paragraph/inference.py --image odia_doc.jpg
```

### Python API

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "shantipriya/odia-ocr-qwen-finetuned_v3").eval()

image = Image.open("odia_doc.jpg").convert("RGB")
prompt = "Extract all Odia text from this image exactly as written, preserving line order and paragraph structure. Return only the Odia text, nothing else."
msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

### With post-processing (recommended for production)

```bash
python inference_with_postprocessing.py --image odia_doc.jpg --checkpoint checkpoint-1300
```

---

## Docker

```bash
docker build -t odia-ocr .
docker run --gpus all -v $(pwd):/workspace odia-ocr \
    python eval.py --checkpoint shantipriya/odia-ocr-qwen-finetuned_v3
```

---

## License

Apache 2.0
