import os, gc, torch
from datasets import load_dataset
from PIL import Image
import editdistance, json

CHECKPOINT = "/root/phase3_paragraph/output_2gpu/checkpoint-300"
GPU_ID     = 0
NEW_IDX    = [10, 50]

def cer(gt, pred):
    gt, pred = gt.strip(), pred.strip()
    return 0.0 if not gt else editdistance.eval(gt, pred) / len(gt)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    CHECKPOINT, torch_dtype=torch.float16,
    device_map={"": GPU_ID}, ignore_mismatched_sizes=True)
model.eval()
# Checkpoint doesn't have processor config — use base model's processor
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(BASE_MODEL)
PROMPT = "Extract all Odia text from this image exactly as written. Output only the extracted text."

print("Loading dataset...")
ds = load_dataset("OdiaGenAIOCR/synthetic_data", split="train")

results = {}
for idx in NEW_IDX:
    row = ds[idx]
    img = row["image"].convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        r = 1024 / max(w, h)
        img = img.resize((int(w*r), int(h*r)), Image.LANCZOS)
    gt = row["extracted_text"].strip()
    # Save to temp file so process_vision_info can handle it reliably
    tmp_img_path = f"/tmp/tmp_indom_{idx}.png"
    img.save(tmp_img_path)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": tmp_img_path},
        {"type": "text",  "text": PROMPT}
    ]}]
    text_in = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(msgs)
    inputs = processor(text=[text_in], images=img_in, padding=True, return_tensors="pt")
    inputs = {k: v.to(f"cuda:{GPU_ID}") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    pred = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    c = cer(gt, pred)
    results[idx] = {"idx": idx, "gt": gt, "pred": pred, "cer": c}
    print(f"idx={idx}: CER={c:.3f}")
    print(f"  gt  = {gt[:100]}")
    print(f"  pred= {pred[:100]}")
    del inputs
    torch.cuda.empty_cache()

print("\nRESULTS_JSON:", json.dumps(results, ensure_ascii=False, indent=2))
with open("/tmp/infer_new_indom_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done.")
