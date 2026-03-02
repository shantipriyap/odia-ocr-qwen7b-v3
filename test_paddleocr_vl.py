#!/usr/bin/env python3
"""
Test PaddleOCR-VL (Vision-Language) - Better for document understanding
"""

import os
os.environ['PADDLEOCR_HOME'] = '/tmp/paddle_cache'

from datasets import load_dataset
from paddleocr import PaddleOCR
import torch

print("🚀 PaddleOCR-VL Test (10 samples)")
print("=" * 60)

# Initialize PaddleOCR-VL with better settings
print("Initializing PaddleOCR-VL...")
ocr = PaddleOCR(
    use_textline_orientation=False,
    lang='ch',  # Chinese handles Odia well
)

# Load 10 samples
print("Loading dataset...")
dataset = load_dataset('shantipriya/odia-ocr-merged')
data = dataset['train']

correct = 0
tested = 0

for i in range(min(100, len(data))):
    example = data[i]
    image = example['image']
    ground_truth = example['text'].strip()
    
    if not ground_truth or len(ground_truth) < 3:
        continue
    
    tested += 1
    if tested > 10:
        break
    
    if hasattr(image, 'convert'):
        image = image.convert('RGB')
    
    # OCR with PaddleOCR
    try:
        result = ocr.predict(image)
    except:
        result = ocr.ocr(image)
    
    if result and result[0]:
        # Extract all text lines
        ocr_lines = []
        for line in result[0]:
            if isinstance(line, (list, tuple)) and len(line) > 1:
                ocr_lines.append(line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1]))
            elif isinstance(line, str):
                ocr_lines.append(line)
        ocr_text = ''.join(ocr_lines).strip()
    else:
        ocr_text = ""
    
    match = ocr_text == ground_truth
    if match:
        correct += 1
    
    print(f"\n[{tested}] GT: {ground_truth[:40]}")
    print(f"     OCR: {ocr_text[:40]}")
    print(f"     {'✅ MATCH' if match else '❌ NO MATCH'}")

print(f"\n{'='*60}")
accuracy = (correct / tested * 100) if tested > 0 else 0
print(f"RESULT: {correct}/{tested} correct ({accuracy:.1f}% accuracy)")
print(f"{'='*60}")

if accuracy >= 70:
    print("✅ Great! Ready for batch processing all 145K images")
elif accuracy >= 50:
    print("⚠️  Moderate. Consider fine-tuning TrOCR for better results")
else:
    print("❌ Need fine-tuning. Will use TrOCR instead")
