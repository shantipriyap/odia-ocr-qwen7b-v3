"""
Full evaluation of Qwen2.5-VL-7B LoRA checkpoint-300:
  1. In-domain samples (OdiaGenAIOCR/synthetic_data): 3 short + 3 long paragraphs
  2. Out-of-domain benchmark (Iftesha/odia-ocr-benchmark): all 151 samples, CER by category
"""
import textwrap, editdistance, json, os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
OCR_PROMPT   = "Please transcribe the Odia text in this image exactly as written, preserving all characters and line breaks."
HF_TOKEN     = "YOUR_HF_TOKEN_HERE"
CHECKPOINT   = "/root/phase3_paragraph/output_2gpu/checkpoint-300"
OUT_FILE     = "/tmp/eval_full_ckpt300_results.json"
MAX_IMG_SIZE = 1024   # resize large images to avoid OOM
GPU_ID       = 0      # CUDA_VISIBLE_DEVICES=1 makes GPU1 appear as device 0


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


def resize_image(img: Image.Image, max_size: int = MAX_IMG_SIZE) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img.convert("RGB")


def run_ocr(model, processor, image: Image.Image, max_new_tokens: int = 512) -> str:
    image = resize_image(image)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": OCR_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    result = processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    torch.cuda.empty_cache()
    return result


# ── Load model once ──────────────────────────────────────────────────────────
print("Loading base model " + BASE_MODEL + " on GPU " + str(GPU_ID) + " ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"":GPU_ID}, token=HF_TOKEN)
print("Loading LoRA adapter from " + CHECKPOINT + " ...")
model = PeftModel.from_pretrained(model, CHECKPOINT)
model.eval()
processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN)


# ── Part 1: In-domain (OdiaGenAIOCR/synthetic_data) ─────────────────────────
print("\n" + "="*60)
print("PART 1: In-domain  (OdiaGenAIOCR/synthetic_data)")
print("="*60)

ds_indomain = load_dataset("OdiaGenAIOCR/synthetic_data", split="train", token=HF_TOKEN)

# Pick 3 short (< 100 chars) and 3 long (> 200 chars) samples
short_samples, long_samples = [], []
for i, s in enumerate(ds_indomain):
    t = s["extracted_text"].strip()
    if len(t) < 100 and len(short_samples) < 3:
        short_samples.append((i, s))
    elif len(t) > 200 and len(long_samples) < 3:
        long_samples.append((i, s))
    if len(short_samples) == 3 and len(long_samples) == 3:
        break

indomain_results = []
for label, group in [("short", short_samples), ("long", long_samples)]:
    print("\n--- In-domain " + label + " paragraphs ---")
    for rank, (idx, sample) in enumerate(group):
        img = sample.get("image")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        gt   = sample["extracted_text"].strip()
        pred = run_ocr(model, processor, img)
        c    = cer(gt, pred)
        indomain_results.append({
            "type": "indomain", "length": label, "idx": idx,
            "gt": gt, "pred": pred, "cer": round(c, 4)
        })
        print("  [" + label + " #" + str(rank+1) + "] CER=" + str(round(c,4)))
        print("  GT  : " + textwrap.shorten(gt,   100))
        print("  PRED: " + textwrap.shorten(pred, 100))


# ── Part 2: Out-of-domain benchmark (Iftesha/odia-ocr-benchmark) ────────────
print("\n" + "="*60)
print("PART 2: Out-of-domain benchmark (Iftesha/odia-ocr-benchmark)")
print("="*60)

ds_bench = load_dataset("Iftesha/odia-ocr-benchmark", split="train", token=HF_TOKEN)
print("Benchmark size: " + str(len(ds_bench)) + " samples")
print("Columns: " + str(ds_bench.column_names))

bench_results = []
cat_stats = {}

for i, sample in enumerate(ds_bench):
    img = sample.get("image")
    if img is None:
        continue
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    gt       = str(sample.get("ground_truth", "")).strip()
    category = str(sample.get("category", "unknown"))
    tlen     = str(sample.get("text_length", "unknown"))
    if not gt:
        continue

    try:
        pred = run_ocr(model, processor, img, max_new_tokens=256)
    except Exception as e:
        print("  ERROR at sample " + str(i) + ": " + str(e))
        torch.cuda.empty_cache()
        pred = ""
    c    = cer(gt, pred)
    bench_results.append({
        "type": "benchmark", "idx": i, "category": category, "text_length": tlen,
        "gt": gt, "pred": pred, "cer": round(c, 4)
    })
    if category not in cat_stats:
        cat_stats[category] = []
    cat_stats[category].append(c)

    if (i+1) % 10 == 0 or (i+1) == 1:
        print("  Processed " + str(i+1) + "/" + str(len(ds_bench)))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK RESULTS BY CATEGORY")
print("="*60)
total_cer_vals = [r["cer"] for r in bench_results]
for cat, vals in sorted(cat_stats.items()):
    avg = sum(vals) / len(vals)
    print("  " + cat.ljust(15) + " n=" + str(len(vals)).rjust(3) +
          "  avg_CER=" + str(round(avg, 4)) +
          "  acc=" + str(round((1-avg)*100, 1)) + "%")

overall_avg = sum(total_cer_vals) / max(len(total_cer_vals), 1)
print("\nOverall benchmark CER : " + str(round(overall_avg, 4)) +
      "  (" + str(round((1-overall_avg)*100, 1)) + "% accuracy)")

# ── Save all results ─────────────────────────────────────────────────────────
all_results = {
    "checkpoint": CHECKPOINT,
    "indomain": indomain_results,
    "benchmark": bench_results,
    "benchmark_summary": {
        cat: {
            "n": len(vals),
            "avg_cer": round(sum(vals)/len(vals), 4),
            "accuracy": round((1 - sum(vals)/len(vals))*100, 1)
        }
        for cat, vals in cat_stats.items()
    },
    "benchmark_overall_cer": round(overall_avg, 4),
    "benchmark_overall_acc": round((1-overall_avg)*100, 1)
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("\nResults saved to " + OUT_FILE)
