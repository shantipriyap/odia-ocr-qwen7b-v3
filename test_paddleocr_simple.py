#!/usr/bin/env python3
import os
os.environ['PADDLEOCR_HOME'] = '/tmp/.paddleocr'

print("🚀 Starting PaddleOCR test...")
print("-" * 70)

# Install
import subprocess
subprocess.run(["pip", "install", "-q", "paddleocr"], capture_output=True)

from paddleocr import PaddleOCR
from datasets import load_dataset
import tempfile
from PIL import Image as PILImage

# Initialize
print("📦 Initializing OCR...")
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

print("📥 Loading dataset...")
dataset = load_dataset("shantipriya/odia-ocr-merged", split="train")
print(f"✅ Dataset loaded: {len(dataset)} samples")

print("\n" + "="*70)
print("🔍 TESTING ON 5 SAMPLES")
print("="*70)

matches = 0
total = 0

for i in range(min(5, len(dataset))):
    try:
        item = dataset[i]
        true_text = item['text'][:80].strip()
        img = item['image']
        
        # Convert image
        if isinstance(img, str):
            img = PILImage.open(img).convert('RGB')
        else:
            img = img.convert('RGB')
        
        # OCR
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            temp_path = tmp.name
            result = ocr.ocr(temp_path, cls=True)
            os.unlink(temp_path)
        
        ocr_text = ' '.join([line[1][0] for line in result[0]]).strip() if result and result[0] else ""
        ocr_text = ocr_text[:80]
        
        is_match = true_text.lower() == ocr_text.lower()
        if is_match:
            matches += 1
        total += 1
        
        print(f"\nSample {i+1}:")
        print(f"  Expected: {true_text}")
        print(f"  OCR:      {ocr_text}")
        print(f"  Result:   {'✓ MATCH' if is_match else '✗ DIFFERENT'}")
    except Exception as e:
        print(f"\nSample {i+1}: ❌ Error - {e}")
        total += 1

print("\n" + "="*70)
print(f"RESULTS: {matches}/{total} samples matched ({100*matches/total:.0f}%)")
print("="*70)

if matches/total > 0.8:
    print("✅ PaddleOCR shows GOOD accuracy!")
elif matches/total > 0.5:
    print("⚠️  PaddleOCR shows MODERATE accuracy - fine-tuning recommended")
else:
    print("❌ PaddleOCR shows LOW accuracy - consider fine-tuning or different model")
