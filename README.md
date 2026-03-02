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
├── train_2gpu.py    # Main training script (GPU0=train, GPU1=eval+push)
├── inference.py     # Inference CLI
└── monitor.py       # Training monitor

eval.py              # Standalone evaluation script
requirements.txt     # Dependencies
```

## Inference

```bash
pip install transformers peft torch pillow accelerate
python phase3_paragraph/inference.py --image odia_doc.jpg
```

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "shantipriya/odia-ocr-qwen7b-phase3").eval()

image = Image.open("odia_doc.jpg").convert("RGB")
prompt = "Extract all Odia text from this image exactly as written, preserving line order and paragraph structure. Return only the Odia text, nothing else."
msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

## License

Apache 2.0
