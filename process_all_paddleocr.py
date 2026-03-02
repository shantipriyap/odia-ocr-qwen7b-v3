#!/usr/bin/env python3
"""
Process All Odia Images with PaddleOCR
Batch process your complete 145K dataset and save results
"""

import os
import sys
import csv
import time
from pathlib import Path
import tempfile

print("=" * 80)
print("🚀 BATCH PROCESS 145K ODIA IMAGES WITH PADDLEOCR")
print("=" * 80)

# Install
print("\n📦 Installing dependencies...")
os.system("pip install -q paddleocr >/dev/null 2>&1")

from paddleocr import PaddleOCR
from datasets import load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

# Initialize
print("🔧 Initializing OCR...")
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

print("📥 Loading dataset...")
dataset = load_dataset("shantipriya/odia-ocr-merged", split="train")
total_samples = len(dataset)
print(f"✓ Dataset ready: {total_samples:,} samples")

# Output file
output_file = Path("odia_ocr_results.csv")
print(f"\n📝 Output: {output_file}")

# Process
print(f"\n📊 Processing {total_samples:,} images...")
print("=" * 80)

start_time = time.time()
success_count = 0
error_count = 0

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['image_id', 'ground_truth', 'ocr_output', 'status', 'confidence'])
    writer.writeheader()
    
    for idx in tqdm(range(total_samples), desc="Processing"):
        try:
            item = dataset[idx]
            ground_truth = item['text']
            img = item['image']
            
            # Convert image
            if isinstance(img, str):
                img = PILImage.open(img).convert('RGB')
            else:
                img = img.convert('RGB')
            
            # OCR
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                result = ocr.ocr(tmp.name, cls=True)
                os.unlink(tmp.name)
            
            # Extract text and confidence
            if result and result[0]:
                ocr_text = ' '.join([line[1][0] for line in result[0]]).strip()
                # Average confidence
                confidence = sum([line[1][1] for line in result[0]]) / len(result[0])
            else:
                ocr_text = ""
                confidence = 0.0
            
            # Write result
            writer.writerow({
                'image_id': idx,
                'ground_truth': ground_truth,
                'ocr_output': ocr_text,
                'status': 'success',
                'confidence': f'{confidence:.3f}'
            })
            
            success_count += 1
            
        except Exception as e:
            try:
                writer.writerow({
                    'image_id': idx,
                    'ground_truth': '',
                    'ocr_output': '',
                    'status': f'error: {str(e)[:50]}',
                    'confidence': 0.0
                })
            except:
                pass
            error_count += 1

elapsed = time.time() - start_time

# Results
print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)
print(f"\nProcessed: {success_count:,} / {total_samples:,} ({100*success_count/total_samples:.1f}%)")
print(f"Errors: {error_count:,}")
print(f"Time: {elapsed/60:.1f} minutes ({elapsed/success_count:.2f}s per image)")
print(f"Output: {output_file}")

# Stats
print("\n📈 Computing statistics...")
matches = 0
try:
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'success':
                if row['ground_truth'].lower() == row['ocr_output'].lower():
                    matches += 1
    
    accuracy = 100 * matches / success_count
    print(f"Exact Match Accuracy: {accuracy:.1f}%")
except:
    print("Could not compute accuracy")

print("\n✅ COMPLETE")
print(f"Results saved to: {output_file}")
