"""
Evaluate Qwen2.5-VL-7B LoRA checkpoint on OdiaGenAIOCR/synthetic_data (paragraph-level OCR).
"""
import textwrap, editdistance
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OCR_PROMPT = "Please transcribe the Odia text in this image exactly as written, preserving all characters and line breaks."
HF_TOKEN   = "YOUR_HF_TOKEN_HERE"
CHECKPOINT = "/root/phase3_paragraph/output_2gpu/checkpoint-300"
N_SAMPLES  = 5


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


def run_ocr(model, processor, image: Image.Image) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": OCR_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


print("Loading base model " + BASE_MODEL + " ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
print("Loading LoRA adapter from " + CHECKPOINT + " ...")
model = PeftModel.from_pretrained(model, CHECKPOINT)
model.eval()
processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN)

print("Loading OdiaGenAIOCR/synthetic_data ...")
ds = load_dataset("OdiaGenAIOCR/synthetic_data", split="train", token=HF_TOKEN)
samples = list(ds.select(range(N_SAMPLES)))

total_cer = 0.0
results = []
for i, sample in enumerate(samples):
    img = sample.get("image")
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    gt   = sample["extracted_text"].strip()
    pred = run_ocr(model, processor, img)
    c    = cer(gt, pred)
    total_cer += c
    results.append((i, gt, pred, c))
    print("\n" + ("─" * 20) + " Sample " + str(i+1) + " " + ("─" * 20))
    print("GT  : " + textwrap.shorten(gt,   120))
    print("PRED: " + textwrap.shorten(pred, 120))
    print("CER : " + str(round(c, 4)))

avg = total_cer / max(len(results), 1)
acc = 1.0 - avg
sep = "=" * 55
print("\n" + sep)
print("Avg CER over " + str(len(results)) + " samples  : " + str(round(avg, 4)))
print("Accuracy (1 - CER)        : " + str(round(acc, 4)) + "  (" + str(round(acc*100, 1)) + " %)")
print(sep)
