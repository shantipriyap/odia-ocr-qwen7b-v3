"""
Quick evaluation of the Qwen2.5-VL-7B LoRA checkpoint on Odia OCR.
Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_qwen_checkpoint.py \
        --checkpoint /root/phase3_paragraph/output_2gpu/checkpoint-100 \
        --n 10
"""
import argparse, textwrap, os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
OCR_PROMPT   = "Please transcribe the Odia text in this image exactly as written, preserving all characters and line breaks."
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

def cer(ref: str, hyp: str) -> float:
    """Simple character error rate."""
    import editdistance
    if not ref:
        return 0.0
    return editdistance.eval(ref, hyp) / max(len(ref), 1)

def load_model(ckpt: str):
    print(f"Loading base model {BASE_MODEL} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    print(f"Loading LoRA adapter from {ckpt} ...")
    model = PeftModel.from_pretrained(model, ckpt)
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    return model, processor

def run_ocr(model, processor, image: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": OCR_PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

def get_gt(sample: dict) -> str:
    for k in ("text", "ground_truth", "transcription", "label", "answer"):
        if k in sample and sample[k]:
            return str(sample[k]).strip()
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/root/phase3_paragraph/output_2gpu/checkpoint-100")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to evaluate")
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint)

    print("Loading Odia OCR dataset ...")
    ds = load_dataset("shantipriya/odia-ocr-merged", split="train", token=HF_TOKEN)
    samples = list(ds.select(range(min(args.n, len(ds)))))

    total_cer = 0.0
    results = []
    for i, sample in enumerate(samples):
        img = sample.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        gt = get_gt(sample)
        pred = run_ocr(model, processor, img)

        try:
            c = cer(gt, pred)
        except ImportError:
            c = float("nan")
        total_cer += c
        results.append((i, gt, pred, c))
        print(f"\n─── Sample {i+1} ───")
        print(f"GT  : {textwrap.shorten(gt,   120)}")
        print(f"PRED: {textwrap.shorten(pred, 120)}")
        print(f"CER : {c:.3f}")

    avg = total_cer / max(len(results), 1)
    print(f"\n{'='*50}")
    print(f"Evaluated {len(results)} samples | Avg CER: {avg:.3f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
