#!/usr/bin/env python3
"""Quick test: 10 samples with PaddleOCR"""

import os
os.environ['PADDLEOCR_HOME'] = '/tmp/paddle_cache'

from datasets import load_dataset
from paddleocr import PaddleOCR
import torch

print("🚀 Quick PaddleOCR Test (10 samples)")
print("=" * 60)

# Check GPU
print(f"GPU: {torch.cuda.is_available()}")

# Initialize
print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, use_gpu=torch.cuda.is_available(), lang='ch')

# Load 10 samples
print("Loading dataset...")
dataset = load_dataset('shantipriya/odia-ocr-merged')
data = dataset['train']

correct = 0
for i in range(min(10, len(data))):
    example = data[i]
    image = example['image']
    ground_truth = example['text'].strip()
    
    if not ground_truth:
        continue
    
    if hasattr(image, 'convert'):
        image = image.convert('RGB')
    
    # OCR
    result = ocr.ocr(image, cls=True)
    if result and result[0]:
        ocr_text = ''.join([line[1][0] for line in result[0]]).strip()
    else:
        ocr_text = ""
    
    match = ocr_text == ground_truth
    if match:
        correct += 1
    
    print(f"\n[{i+1}] Ground: {ground_truth[:30]}")
    print(f"    OCR:    {ocr_text[:30]}")
    print(f"    Match:  {'✅' if match else '❌'}")

print(f"\n{'='*60}")
print(f"RESULT: {correct}/10 correct ({correct*10}% accuracy)")
print(f"{'='*60}")
