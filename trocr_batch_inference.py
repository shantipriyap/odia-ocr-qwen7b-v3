#!/usr/bin/env python3
"""
TrOCR Batch Inference - Process all 145K Odia images
Run after fine-tuning is complete
"""

import torch
import json
import csv
from pathlib import Path
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from tqdm import tqdm
from datetime import datetime
import os

print("=" * 80)
print("🚀 TrOCR BATCH INFERENCE - 145K ODIA IMAGES")
print("=" * 80)

# Paths
model_dir = "./trocr-odia-finetuned"
dataset_name = "shantipriya/odia-ocr-merged"
output_csv = "trocr_full_results.csv"
checkpoint_json = "trocr_inference_checkpoint.json"

# Check model exists
if not Path(model_dir).exists():
    print(f"❌ Model not found at: {model_dir}")
    print(f"   Please run: python3 trocr_finetuning_a100_optimized.py first")
    exit(1)

print(f"✅ Model found at: {model_dir}")

# Load model
print(f"\n📦 Loading model...")
try:
    processor = ViTImageProcessor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
except Exception as e:
    print(f"⚠️  Could not load from fine-tuned dir, trying base model...")
    model_base = "microsoft/trocr-base-stage1"
    processor = ViTImageProcessor.from_pretrained(model_base)
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')
    print(f"✅ Model on GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
else:
    print(f"⚠️  Running on CPU (very slow!)")

# Load dataset
print(f"\n📥 Loading dataset...")
dataset = load_dataset(dataset_name)
data = dataset['train']
print(f"✅ {len(data)} samples loaded")

# Load checkpoint if exists
start_idx = 0
if os.path.exists(checkpoint_json):
    with open(checkpoint_json, 'r') as f:
        checkpoint = json.load(f)
        start_idx = checkpoint.get('last_index', 0)
        print(f"\n🔄 Resuming from index {start_idx}")

# Initialize CSV if new
if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'ground_truth', 'ocr_output', 'exact_match', 'confidence', 'error'])

# Generation config
gen_config = GenerationConfig.from_pretrained(model_dir) if Path(f"{model_dir}/generation_config.json").exists() else None
if gen_config:
    gen_config.max_length = 128
else:
    # Default config
    gen_config = GenerationConfig(
        max_length=128,
        num_beams=1,
        temperature=0.8,
    )

# Inference
print(f"\n🔄 Processing images...")
print("-" * 80)

correct_count = 0
total_processed = 0
errors = 0

with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    for idx in tqdm(range(start_idx, len(data)), initial=start_idx, total=len(data), desc="Inference"):
        try:
            example = data[idx]
            image = example['image']
            ground_truth = example['text'].strip()
            
            # Skip if no text
            if not ground_truth or len(ground_truth) < 1:
                continue
            
            # Prepare image
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # Process image
            try:
                pixel_values = processor(image, return_tensors='pt').pixel_values
                if torch.cuda.is_available():
                    pixel_values = pixel_values.to('cuda')
                
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values, generation_config=gen_config)
                
                predicted = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Estimate confidence (no explicit confidence from model, but we can use presence)
                confidence = 1.0 if predicted else 0.0
                
            except Exception as e:
                predicted = ""
                confidence = 0.0
            
            # Check if exact match
            exact_match = (predicted == ground_truth)
            if exact_match:
                correct_count += 1
            
            total_processed += 1
            
            # Write result
            writer.writerow([
                idx,
                ground_truth,
                predicted,
                1 if exact_match else 0,
                f"{confidence:.3f}",
                ""
            ])
            
        except Exception as e:
            errors += 1
            try:
                writer.writerow([idx, "", "", 0, "0.0", str(e)[:50]])
            except:
                pass
        
        # Save checkpoint every 500 samples
        if (idx + 1) % 500 == 0:
            with open(checkpoint_json, 'w') as cp:
                json.dump({
                    'last_index': idx + 1,
                    'timestamp': datetime.now().isoformat(),
                    'correct_so_far': correct_count,
                    'total_processed': total_processed,
                    'current_accuracy': (correct_count / total_processed * 100) if total_processed > 0 else 0,
                }, cp)
            
            if total_processed > 0:
                current_acc = (correct_count / total_processed) * 100
                eta_secs = (len(data) - idx - 1) * (idx + 1 - start_idx) / (idx + 1 - start_idx) if (idx + 1) > start_idx else 0
                tqdm.write(f"   Progress: {total_processed} processed, {current_acc:.1f}% accuracy")

print(f"\n" + "=" * 80)
print(f"✅ INFERENCE COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"=" * 80)

# Compute final metrics
if total_processed > 0:
    accuracy = (correct_count / total_processed) * 100
    print(f"\n📊 Final Results:")
    print(f"   Total processed: {total_processed:,}")
    print(f"   Correct matches: {correct_count:,}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Errors: {errors:,}")
    
    if accuracy >= 85:
        print(f"\n✅ Excellent accuracy! Model is production-ready")
    elif accuracy >= 70:
        print(f"\n⚠️  Good accuracy. Consider fine-tuning more epochs for better results")
    else:
        print(f"\n⚠️  Accuracy could be better. May need longer training")
else:
    print(f"⚠️  No samples processed!")

# Create summary
summary = {
    'total_processed': total_processed,
    'correct_count': correct_count,
    'accuracy': (correct_count / total_processed * 100) if total_processed > 0 else 0,
    'errors': errors,
    'timestamp': datetime.now().isoformat(),
    'model': model_dir,
    'dataset': dataset_name,
    'output_csv': output_csv,
}

with open('trocr_full_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n📁 Results saved:")
print(f"   CSV: {output_csv}")
print(f"   Summary: trocr_full_summary.json")

print(f"\n" + "=" * 80)
