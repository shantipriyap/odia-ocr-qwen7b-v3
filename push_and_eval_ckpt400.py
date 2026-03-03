#!/usr/bin/env python3
"""
Push checkpoint-400 to HF and run quick eval (15 benchmark samples),
then save results to /tmp/eval_ckpt400_results.json
"""
import os, io, gc, json, torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import HfApi
import editdistance

HF_TOKEN   = "YOUR_HF_TOKEN_HERE"
MODEL_REPO = "shantipriya/odia-ocr-qwen-finetuned_v3"
CHECKPOINT = "/root/phase3_paragraph/output_2gpu/checkpoint-400"
GPU_ID     = 0   # CUDA_VISIBLE_DEVICES=1 → logical 0

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def cer(gt, pred):
    gt, pred = gt.strip(), pred.strip()
    return 0.0 if not gt else editdistance.eval(gt, pred) / len(gt)

# ── STEP 1: Push checkpoint-400 ───────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Pushing checkpoint-400 to HF...")
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path=CHECKPOINT,
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Add checkpoint-400 (loss=0.072, step 400/3000)",
    ignore_patterns=["optimizer.pt", "rng_state*.pth", "scaler.pt"],
)
print("  checkpoint-400 pushed OK")

# ── STEP 2: Quick eval on Iftesha benchmark (15 samples, 5 per category) ─────
print("\n" + "=" * 60)
print("STEP 2: Quick eval on Iftesha/odia-ocr-benchmark (15 samples)...")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print("Loading model from checkpoint-400...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.float16,
    device_map={"": GPU_ID},
    ignore_mismatched_sizes=True,
)
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
PROMPT = "Extract all Odia text from this image exactly as written. Output only the extracted text."

print("Loading benchmark dataset...")
bench_ds = load_dataset("Iftesha/odia-ocr-benchmark", split="train")

# Pick 3 from scene_text (best category) + 3 handwritten + 3 printed + 3 digital + 3 newspaper
EVAL_INDICES = [
    120, 121, 122,   # scene_text (good performers)
    39, 40, 41,      # handwritten
    53, 54, 55,      # printed
    5, 6, 7,         # Digital
    24, 25, 26,      # Newspaper
]

results = []
for idx in EVAL_INDICES:
    row = bench_ds[idx]
    img = row["image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else Image.fromarray(img)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        r = 1024 / max(w, h)
        img = img.resize((int(w*r), int(h*r)), Image.LANCZOS)

    tmp_path = f"/tmp/eval400_bench_{idx}.png"
    img.save(tmp_path)
    gt = row.get("ground_truth", row.get("text", "")).strip()
    cat = row.get("category", "unknown")

    msgs = [{"role": "user", "content": [
        {"type": "image", "image": tmp_path},
        {"type": "text",  "text": PROMPT},
    ]}]
    text_in = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(msgs)
    inputs = processor(text=[text_in], images=img_in, padding=True, return_tensors="pt")
    inputs = {k: v.to(f"cuda:{GPU_ID}") for k, v in inputs.items()}
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        pred = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        pred = f"ERROR: {e}"
    c = cer(gt, pred)
    results.append({"idx": idx, "category": cat, "gt": gt, "pred": pred, "cer": c})
    print(f"  idx={idx:3d} cat={cat:12s} CER={c:.3f}  gt={gt[:40]:40s}")
    del inputs
    torch.cuda.empty_cache()
    gc.collect()

del model, processor
gc.collect()
torch.cuda.empty_cache()

# Compute overall and per-category stats
from collections import defaultdict
by_cat = defaultdict(list)
for r in results:
    by_cat[r["category"]].append(r["cer"])

overall_cer = sum(r["cer"] for r in results) / len(results)
overall_acc = (1 - overall_cer) * 100

print("\n--- Results ---")
for cat, cers in by_cat.items():
    avg = sum(cers) / len(cers)
    print(f"  {cat}: n={len(cers)}, avg_cer={avg:.3f}, acc={100*(1-avg):.1f}%")
print(f"  OVERALL: n={len(results)}, avg_cer={overall_cer:.3f}, acc={overall_acc:.1f}%")

output = {
    "checkpoint": "checkpoint-400",
    "loss": 0.07223942279815673,
    "results": results,
    "by_category": {cat: {"n": len(cers), "avg_cer": sum(cers)/len(cers), "acc": 100*(1-sum(cers)/len(cers))} for cat, cers in by_cat.items()},
    "overall_cer": overall_cer,
    "overall_acc": overall_acc,
}
with open("/tmp/eval_ckpt400_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("\nResults saved to /tmp/eval_ckpt400_results.json")
print("Done.")
