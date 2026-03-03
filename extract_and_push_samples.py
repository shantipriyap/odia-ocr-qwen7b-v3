#!/usr/bin/env python3
"""
Extract representative sample images from both datasets,
run inference on 2 extra in-domain samples, upload all to HF model repo,
then print the full README section.
"""

import os, io, json, textwrap, gc, torch
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from huggingface_hub import HfApi

HF_TOKEN     = "YOUR_HF_TOKEN_HERE"
MODEL_REPO   = "shantipriya/odia-ocr-qwen-finetuned_v3"
CHECKPOINT   = "/root/phase3_paragraph/output_2gpu/checkpoint-300"
RESULTS_JSON = "/tmp/eval_full_ckpt300_results.json"
OUT_DIR      = Path("/tmp/ocr_sample_images")
OUT_DIR.mkdir(exist_ok=True)

GPU_ID = 0   # CUDA_VISIBLE_DEVICES=1 → logical device 0
MAX_IMG = 800

api = HfApi(token=HF_TOKEN)

with open(RESULTS_JSON) as f:
    data = json.load(f)
bench_results = {s["idx"]: s for s in data["benchmark"]}
indomain_eval = {r["idx"]: r for r in data["indomain"]}  # keys: 0,1,2

BENCH_SEL = {
    "printed":    {"good": 53,  "mixed": 94,  "bad":  93},
    "scene_text": {"good": 120, "mixed": 140, "bad": 138},
    "handwritten":{"good": 39,  "mixed": 40,  "bad":  50},
    "Digital":    {"good": 5,   "mixed": 4,   "bad":   0},
    "Book":       {"good": 17,  "mixed": 15,  "bad":  19},
    "Newspaper":  {"good": 24,  "mixed": 30,  "bad":  27},
}

INDOMAIN_IDX = [0, 1, 2, 10, 50]
NEW_INDOMAIN  = [10, 50]

def resize_save(img, name, max_size=MAX_IMG):
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else Image.fromarray(img)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w*r), int(h*r)), Image.LANCZOS)
    path = OUT_DIR / name
    img.save(path, "PNG")
    return path

def upload_file(local, repo_path):
    api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                    repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)
    print(f"  uploaded {repo_path}")

def cer_compute(gt, pred):
    import editdistance
    gt, pred = gt.strip(), pred.strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(gt, pred) / len(gt)

def quality(cer):
    if cer < 0.4:   return "Good"
    elif cer < 0.8: return "Mixed"
    else:           return "Bad"

def wrap(t, w=55):
    return "<br>".join(textwrap.wrap(str(t)[:200], w))

# ── STEP 1: benchmark images ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Benchmark images")
bench_idx_set = {idx for sel in BENCH_SEL.values() for idx in sel.values()}
print(f"Loading Iftesha/odia-ocr-benchmark ({len(bench_idx_set)} images)...")
bench_ds = load_dataset("Iftesha/odia-ocr-benchmark", split="train")
for idx in sorted(bench_idx_set):
    row = bench_ds[idx]
    upload_file(resize_save(row["image"], f"bench_{idx}.png"),
                f"assets/samples/bench_{idx}.png")
del bench_ds; gc.collect()

# ── STEP 2: in-domain images + inference on new ones ────────────────────────
print("\n" + "=" * 60)
print("STEP 2: In-domain images + new inference")
print("Loading OdiaGenAIOCR/synthetic_data...")
indomain_ds = load_dataset("OdiaGenAIOCR/synthetic_data", split="train")

for idx in INDOMAIN_IDX:
    upload_file(resize_save(indomain_ds[idx]["image"], f"indomain_{idx}.png"),
                f"assets/samples/indomain_{idx}.png")

new_results = {}
print(f"\nRunning Qwen inference on new indices {NEW_INDOMAIN}...")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    CHECKPOINT, torch_dtype=torch.float16, device_map={"": GPU_ID})
model.eval()
processor = AutoProcessor.from_pretrained(CHECKPOINT)
PROMPT = "Extract all Odia text from this image exactly as written. Output only the extracted text."

for idx in NEW_INDOMAIN:
    row = indomain_ds[idx]
    img = row["image"].convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        rat = 1024 / max(w, h)
        img = img.resize((int(w*rat), int(h*rat)), Image.LANCZOS)
    gt = row["extracted_text"].strip()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": PROMPT},
    ]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_in], images=img_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(f"cuda:{GPU_ID}") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    pred = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    cer = cer_compute(gt, pred)
    new_results[idx] = {"idx": idx, "gt": gt, "pred": pred, "cer": cer}
    print(f"  idx={idx}: CER={cer:.3f} | gt={gt[:60]}")
    print(f"            pred={pred[:60]}")
    del inputs; torch.cuda.empty_cache()

del model, processor, indomain_ds; gc.collect(); torch.cuda.empty_cache()

# ── STEP 3: build README section ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building README section...")
BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main/assets/samples"

lines = []
lines.append("## 📸 Sample Inference Results — Checkpoint 300\n")
lines.append("> **Out-of-domain** (`Iftesha/odia-ocr-benchmark`): 6 categories × 3 quality levels (18 samples total)  ")
lines.append("> **In-domain** (`OdiaGenAIOCR/synthetic_data`): 5 synthesised paragraph images  ")
lines.append("> Quality: ✅ Good (CER < 0.4) · ⚠️ Mixed (CER 0.4–0.8) · ❌ Bad (CER > 0.8)\n")
lines.append("---\n")

lines.append("### 🔤 Out-of-Domain Benchmark Samples\n")
for cat, sel in BENCH_SEL.items():
    lines.append(f"<details>")
    lines.append(f"<summary><b>{cat}</b> — click to expand</summary>\n")
    lines.append("| Quality | Image | Ground Truth | Predicted | CER |")
    lines.append("|:-------:|:-----:|:------------|:----------|:---:|")
    for q_label, idx in [("Good", sel["good"]), ("Mixed", sel["mixed"]), ("Bad", sel["bad"])]:
        r = bench_results[idx]
        img_md = f'<img src="{BASE}/bench_{idx}.png" width="200"/>'
        cer_val = r["cer"]
        emoji = "✅" if quality(cer_val) == "Good" else ("⚠️" if quality(cer_val) == "Mixed" else "❌")
        lines.append(f"| {emoji} {quality(cer_val)} | {img_md} | {wrap(r['gt'])} | {wrap(r['pred'])} | **{cer_val:.3f}** |")
    lines.append("\n</details>\n")

lines.append("\n---\n")
lines.append("### 📄 In-Domain Samples (`OdiaGenAIOCR/synthetic_data`)\n")
lines.append("| # | Image | Ground Truth | Predicted | CER | Quality |")
lines.append("|:-:|:-----:|:------------|:----------|:---:|:-------:|")

all_indomain = {**{k: v for k, v in indomain_eval.items()}, **new_results}
for i, idx in enumerate(INDOMAIN_IDX, 1):
    r = all_indomain.get(idx, {"idx": idx, "gt": "", "pred": "—", "cer": None})
    img_md = f'<img src="{BASE}/indomain_{idx}.png" width="200"/>'
    cer_val = r.get("cer")
    cer_str = f"**{cer_val:.3f}**" if cer_val is not None else "—"
    q_str = quality(cer_val) if cer_val is not None else "—"
    emoji = "✅" if q_str == "Good" else ("⚠️" if q_str == "Mixed" else ("❌" if q_str == "Bad" else ""))
    lines.append(f"| {i} | {img_md} | {wrap(r.get('gt', ''))} | {wrap(r.get('pred', ''))} | {cer_str} | {emoji} {q_str} |")

section = "\n".join(lines)
with open("/tmp/readme_samples_section.txt", "w") as f:
    f.write(section)
print("\n[README section saved to /tmp/readme_samples_section.txt]")
print("✅ Done.")
