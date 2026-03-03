#!/usr/bin/env python3
"""
PaddleOCR Batch Processor - Process all 145K Odia OCR images
Computes accuracy and saves results incrementally
"""

import os
import json
import csv
from pathlib import Path
from datasets import load_dataset
from paddleocr import PaddleOCR
import torch
from tqdm import tqdm
from datetime import datetime

print("=" * 80)
print("🚀 PaddleOCR BATCH PROCESSOR - ODIA OCR DATASET")
print("=" * 80)

# Check GPU availability
print(f"🔧 GPU Available: {torch.cuda.is_available()}")
print(f"📦 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Initialize PaddleOCR for Odia
print("\n📥 Initializing PaddleOCR (downloading models if needed)...")
ocr = PaddleOCR(
    use_angle_cls=True,
    use_gpu=torch.cuda.is_available(),
    lang='ch'  # Chinese model works well for Odia script
)
print("✅ PaddleOCR ready!\n")

# Load dataset
print("📂 Loading dataset...")
dataset = load_dataset('shantipriya/odia-ocr-merged')
train_data = dataset['train']
print(f"✅ Dataset loaded: {len(train_data)} samples\n")

# Output file
output_csv = 'paddleocr_results.csv'
checkpoint_json = 'paddleocr_checkpoint.json'

# Load checkpoint if exists
start_idx = 0
if os.path.exists(checkpoint_json):
    with open(checkpoint_json, 'r') as f:
        checkpoint = json.load(f)
        start_idx = checkpoint.get('last_index', 0)
        print(f"🔄 Resuming from index {start_idx}\n")

# Initialize CSV if new
if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'image_id', 'ground_truth', 'ocr_output', 'exact_match', 'confidence', 'error'])

# Process images
correct_count = 0
total_processed = 0
errors = 0

print("🔄 Processing images...")
print("-" * 80)

with open(output_csv, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    for idx in tqdm(range(start_idx, len(train_data)), initial=start_idx, total=len(train_data), desc="Processing"):
        try:
            example = train_data[idx]
            image = example.get('image')
            ground_truth = example.get('text', '').strip()
            
            # Skip if no text
            if not ground_truth:
                continue
            
            # Convert image if needed
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # Run OCR
            try:
                result = ocr.ocr(image, cls=True)
                
                # Extract text from result
                if result and result[0]:
                    ocr_output = ''.join([line[1][0] for line in result[0]])
                else:
                    ocr_output = ''
                
                # Compute confidence (average of all line confidences)
                conf = 0.0
                if result and result[0]:
                    confs = [line[1][1] for line in result[0]]
                    conf = sum(confs) / len(confs) if confs else 0
                
                # Check if exact match
                exact_match = (ocr_output.strip() == ground_truth)
                if exact_match:
                    correct_count += 1
                
                total_processed += 1
                
                # Write result
                writer.writerow([
                    idx,
                    f"odia_ocr_{idx:06d}",
                    ground_truth,
                    ocr_output,
                    1 if exact_match else 0,
                    f"{conf:.3f}",
                    ""
                ])
                
            except Exception as e:
                errors += 1
                writer.writerow([
                    idx,
                    f"odia_ocr_{idx:06d}",
                    ground_truth,
                    "",
                    0,
                    "0.0",
                    str(e)[:100]
                ])
        
        except Exception as e:
            errors += 1
            continue
        
        # Save checkpoint every 500 samples
        if (idx + 1) % 500 == 0:
            with open(checkpoint_json, 'w') as cp:
                json.dump({
                    'last_index': idx + 1,
                    'timestamp': datetime.now().isoformat(),
                    'correct_so_far': correct_count,
                    'total_processed': total_processed
                }, cp)
            
            if total_processed > 0:
                accuracy = (correct_count / total_processed) * 100
                print(f"\n📊 Progress: {total_processed} processed, Accuracy: {accuracy:.2f}%")

print("\n" + "=" * 80)
print("✅ PROCESSING COMPLETE")
print("=" * 80)

# Compute final metrics
if total_processed > 0:
    accuracy = (correct_count / total_processed) * 100
    print(f"\n📊 Final Results:")
    print(f"   Total processed: {total_processed}")
    print(f"   Correct matches: {correct_count}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Errors: {errors}")
    print(f"\n📁 Results saved to: {output_csv}")
else:
    print("⚠️  No samples processed!")

# Create summary
summary = {
    'total_processed': total_processed,
    'correct_count': correct_count,
    'accuracy': (correct_count / total_processed * 100) if total_processed > 0 else 0,
    'errors': errors,
    'timestamp': datetime.now().isoformat(),
    'model': 'PaddleOCR',
    'dataset': 'shantipriya/odia-ocr-merged'
}

with open('paddleocr_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n📊 Summary saved to: paddleocr_summary.json")
