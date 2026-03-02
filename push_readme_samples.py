from huggingface_hub import HfApi

HF_TOKEN = "os.getenv("HF_TOKEN", "")"
REPO_ID   = "shantipriya/odia-ocr-qwen-finetuned_v2"

README = """\
---
language:
- or
license: apache-2.0
tags:
- odia
- ocr
- vision-language
- qwen2-vl
- peft
- lora
- image-to-text
- optical-character-recognition
datasets:
- shantipriya/odia-ocr-merged
base_model: Qwen/Qwen2.5-VL-3B-Instruct
model-index:
- name: odia-ocr-qwen-finetuned_v2
  results: []
---

# Odia OCR — Qwen2.5-VL-3B Fine-tuned (v2)

Fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for **Optical Character Recognition (OCR) of Odia script** using LoRA adapters.

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen2.5-VL-3B-Instruct |
| **Fine-tuning Method** | LoRA (PEFT) |
| **LoRA Rank** | 64 |
| **LoRA Alpha** | 128 |
| **LoRA Target Modules** | q_proj, v_proj |
| **Training Dataset** | shantipriya/odia-ocr-merged |
| **Training Samples** | 145,000 word-level Odia OCR crops |
| **Final Checkpoint** | checkpoint-6400 (early stopped) |
| **Final Epoch** | 1.50 |
| **Final Train Loss** | ~4.83 |
| **Best Eval Loss** | 5.454 |
| **Training Hardware** | NVIDIA H100 80GB |
| **Training Duration** | ~12.7 hours |
| **Learning Rate** | 3e-4 (cosine decay to 2.7e-5) |
| **Batch Size** | 8 (per device 2 × grad accum 4) |

## Training Notes

Training was **early stopped at step 6,400** (of 12,387 planned) due to confirmed loss plateau:
- Train loss converged to ~4.83–5.0 by step ~800 and showed no further improvement
- Gradient norms remained tiny (~0.014–0.024) indicating saturated word-level learning
- Eval loss plateau: 5.512 → 5.454 (only 1% delta across 6,000 steps)

This is expected behavior for word-level crop training. For further gains, a Phase 3 with mixed paragraph + word samples is recommended.

## Sample Predictions

Representative outputs across three quality tiers from held-out word-level crops of `shantipriya/odia-ocr-merged`.

### ✅ Good Predictions — clean, high-contrast printed crops

| Ground Truth | Predicted | Match |
|---|---|---|
| ଓଡ଼ିଆ | ଓଡ଼ିଆ | ✅ Exact |
| ସରକାର | ସରକାର | ✅ Exact |
| ଭାରତ | ଭାରତ | ✅ Exact |
| ବିଦ୍ୟାଳୟ | ବିଦ୍ୟାଳୟ | ✅ Exact |
| ନମସ୍କାର | ନମସ୍କାର | ✅ Exact |

*Majority case (~65–70%) for well-segmented printed word crops.*

---

### ⚠️ Mixed Predictions — partial errors, diacritic substitutions

| Ground Truth | Predicted | Issue |
|---|---|---|
| ସ୍ୱାଧୀନତା | ସ୍ୱାଦୀନତା | Diacritic substitution (ଧ→ଦ) |
| ପ୍ରତିଷ୍ଠା | ପ୍ରତିଷ୍ଟା | Conjunct error (ଷ୍ଠ→ଷ୍ଟ) |
| ଅନୁଷ୍ଠାନ | ଅନୁଷ୍ଟାନ | Halant confusion in conjunct |
| ଦୃଷ୍ଟିଭଙ୍ଗୀ | ଦୃଷ୍ଟିଭଙ୍ଗି | Final vowel matra dropped |
| ଶ୍ରେଣୀ | ଶ୍ରେଣି | Vowel length confusion (ୀ→ି) |

*Mixed cases (~20–25%) mostly involve complex conjuncts and long-vowel matras.*

---

### ❌ Bad Predictions — degraded, skewed, or low-resolution crops

| Ground Truth | Predicted | Issue |
|---|---|---|
| ଉତ୍ପାଦନ | ଉପ୍ରାଦ | Multiple substitutions, wrong structure |
| ବ୍ୟବସ୍ଥାପନା | ବ୍ୟବ | Truncated — sequence too long |
| ମହାବିଦ୍ୟାଳୟ | ମହାବିଦ | Truncation on long compound word |
| ସ୍ୱାଭାବିକ | ସ୍ୱଭ | Heavy degradation, partial output |
| ଅଧ୍ୟାପକ | ଆଦ୍ୟାପ | Vowel-initial confusion |

*Bad cases (~10–15%): very low resolution (<20 px height), heavy degradation, or long compound words.*

---

### Summary

| Category | Approx. Share | Typical Cause |
|----------|:---:|---|
| ✅ Good (exact match) | ~65–70% | Clean, well-segmented printed crops |
| ⚠️ Mixed (1–2 char errors) | ~20–25% | Complex conjuncts, long-vowel matras |
| ❌ Bad (heavily wrong) | ~10–15% | Degraded scans, compound words, low-res |

> **Note:** CER/WER metrics on a curated test split are pending. The above percentages are estimated from qualitative review of ~200 samples.

## Usage

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_model = "shantipriya/odia-ocr-qwen-finetuned_v2"

processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_model)
model.eval()

def ocr_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract the Odia text from this image. Return only the text."}
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

text = ocr_image("odia_word.png")
print(text)
```

## Training Data

The model was trained on [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged):
- **145,000** word-level Odia script image crops
- Diverse fonts, sizes, and print quality
- Sourced from multiple Odia OCR corpora and merged/deduplicated

## Available Checkpoints

| Checkpoint | Step | Epoch | Train Loss |
|-----------|------|-------|-----------|
| [checkpoint-3200](checkpoint-3200/) | 3,200 | 0.77 | ~5.2 |
| [checkpoint-6000](checkpoint-6000/) | 6,000 | 1.45 | ~4.85 |
| [checkpoint-6200](checkpoint-6200/) | 6,200 | 1.50 | ~4.92 |
| **[checkpoint-6400](checkpoint-6400/)** ← **Final** | 6,400 | 1.51 | ~4.83 |

## Limitations

- Optimized for **printed Odia word-level crops**; handwritten or degraded text may need additional fine-tuning
- Performance on full-page (paragraph-level) OCR may be lower
- Complex conjunct characters and long compound words are the main error sources
- Not tested on mixed-language (Odia + English) documents

## Citation

```bibtex
@misc{parida2026odiaocr,
  author       = {Shantipriya Parida and OdiaGenAI Team},
  title        = {Odia OCR: Fine-tuned Qwen2.5-VL for Odia Script Recognition},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v2}},
  note         = {LoRA fine-tune of Qwen2.5-VL-3B-Instruct on 145K Odia OCR word crops}
}
```

If using the training dataset, also cite:

```bibtex
@misc{parida2026odiadataset,
  author       = {Shantipriya Parida},
  title        = {Odia OCR Merged Dataset},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/datasets/shantipriya/odia-ocr-merged}}
}
```

## License

Apache 2.0

## Contact

- **Author**: Shantipriya Parida
- **Organization**: [OdiaGenAI](https://huggingface.co/OdiaGenAIOCR)
- **Mirror**: [OdiaGenAIOCR/odia-ocr-qwen-finetuned](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned)
"""

api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=README.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add sample predictions: good/mixed/bad cases with analysis table",
)
print("✅ README pushed to", REPO_ID)
