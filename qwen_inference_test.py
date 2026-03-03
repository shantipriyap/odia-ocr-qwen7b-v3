#!/usr/bin/env python3
"""
Inference test on Qwen2.5-VL with ODIA OCR dataset
"""
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🔍 QWEN2.5-VL INFERENCE TEST (ODIA OCR)")
print("="*70)

# Load model
print("\n1️⃣ Loading model...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("   ✅ Model loaded")

# Load dataset
print("\n2️⃣ Loading dataset...")
ds = load_dataset("shantipriya/odia-ocr-merged")['train'].select(range(10))
print(f"   ✅ {len(ds)} samples selected")

# Run inference
print("\n3️⃣ Running inference...")
print("-" * 70)

correct = 0
total = 0

for idx, sample in enumerate(ds):
    try:
        # Get image and ground truth
        img = sample['image']
        gt_text = sample['text']
        
        if not gt_text:
            continue
        
        # Convert to PIL Image if needed
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img)).convert('RGB')
        elif not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')
        else:
            img = img.convert('RGB')
        
        # Format message
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Read the text in this image. Respond with ONLY the text, nothing else."}
            ]}
        ]
        
        # Process
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=text, images=img, return_tensors="pt")
        
        # Move to model device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                top_p=0.9
            )
        
        # Decode
        pred_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (after "assistant\n")
        if "assistant\n" in pred_text:
            pred_text = pred_text.split("assistant\n")[-1].strip()
        else:
            pred_text = pred_text.strip()
        
        # Compare
        is_correct = pred_text.lower() == gt_text.lower()
        correct += int(is_correct)
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Sample {idx+1}:")
        print(f"   GT:   '{gt_text}'")
        print(f"   Pred: '{pred_text}'")
        print()
        
    except Exception as e:
        print(f"❌ Error on sample {idx+1}: {e}")
        print()

print("-" * 70)
print(f"\n📊 RESULTS:")
if total > 0:
    print(f"   Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
else:
    print(f"   No samples processed")
print(f"   Samples processed: {total}")
print("\n✅ Inference test complete!")
print("="*70)
