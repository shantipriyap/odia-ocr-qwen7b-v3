#!/usr/bin/env python3
"""
Evaluate fine-tuned TrOCR model on Odia OCR dataset
Computes CER, WER, and exact match accuracy
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
import numpy as np
from datetime import datetime

def cer(ref, hyp):
    """Character Error Rate"""
    if len(ref) == 0:
        return 0 if len(hyp) == 0 else 1
    
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(ref)][len(hyp)] / len(ref)

def wer(ref, hyp):
    """Word Error Rate"""
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if len(ref_words) == 0:
        return 0 if len(hyp_words) == 0 else 1
    
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

print("=" * 80)
print("🎯 TrOCR FINE-TUNED MODEL EVALUATION")
print("=" * 80)

# Paths
model_dir = "./trocr-odia-finetuned"
dataset_name = "shantipriya/odia-ocr-merged"
output_csv = "trocr_evaluation_results.csv"

# Check model exists
if not Path(model_dir).exists():
    print(f"❌ Model not found at: {model_dir}")
    print(f"   Please run: python3 trocr_finetuning_a100_optimized.py")
    exit(1)

# Load model
print(f"\n📦 Loading fine-tuned model...")
try:
    processor = ViTImageProcessor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
except:
    # Fallback to base model if fine-tuned config not available
    model_base = "microsoft/trocr-base-stage1"
    processor = ViTImageProcessor.from_pretrained(model_base)
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')
    print(f"✅ Model on GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"⚠️  Running on CPU (slow)")

# Load dataset
print(f"\n📥 Loading dataset...")
dataset = load_dataset(dataset_name)
data = dataset['train']
print(f"✅ {len(data)} samples loaded")

# Generation config
gen_config = GenerationConfig.from_pretrained(model_dir)
gen_config.max_length = 128
gen_config.num_beams = 4

# Evaluate
print(f"\n🔄 Evaluating on first 500 samples...")
print("-" * 80)

results = []
total_cer = 0
total_wer = 0
exact_matches = 0
sample_count = 0
errors = 0

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['index', 'ground_truth', 'predicted', 'cer', 'wer', 'exact_match', 'error'])
    
    for idx in tqdm(range(min(500, len(data)))):
        try:
            example = data[idx]
            image = example['image']
            ground_truth = example['text'].strip()
            
            if not ground_truth or len(ground_truth) < 1:
                continue
            
            # Prepare image
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # Process and generate
            pixel_values = processor(image, return_tensors='pt').pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, generation_config=gen_config)
            
            predicted = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Compute metrics
            char_error = cer(ground_truth, predicted)
            word_error = wer(ground_truth, predicted)
            is_exact_match = (predicted == ground_truth)
            
            total_cer += char_error
            total_wer += word_error
            if is_exact_match:
                exact_matches += 1
            
            sample_count += 1
            
            writer.writerow([
                idx,
                ground_truth,
                predicted,
                f"{char_error:.3f}",
                f"{word_error:.3f}",
                1 if is_exact_match else 0,
                ""
            ])
            
        except Exception as e:
            errors += 1
            writer.writerow([idx, "", "", "", "", 0, str(e)[:50]])

# Print results
print(f"\n" + "=" * 80)
print(f"📊 EVALUATION RESULTS")
print(f"=" * 80)

if sample_count > 0:
    avg_cer = total_cer / sample_count
    avg_wer = total_wer / sample_count
    accuracy = (exact_matches / sample_count) * 100
    
    print(f"\nSamples evaluated: {sample_count}")
    print(f"Exact matches: {exact_matches}/{sample_count} ({accuracy:.1f}%)")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Errors: {errors}")
    
    if accuracy >= 80:
        print(f"\n✅ Excellent! Ready for production")
    elif accuracy >= 60:
        print(f"\n⚠️  Good results. Consider fine-tuning more epochs")
    else:
        print(f"\n⚠️  Need improvement. Try longer training")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'samples_evaluated': sample_count,
        'exact_matches': exact_matches,
        'accuracy': accuracy,
        'avg_cer': avg_cer,
        'avg_wer': avg_wer,
        'errors': errors,
        'model_dir': model_dir,
        'dataset': dataset_name,
    }
    
    with open('trocr_evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_csv}")
    print(f"   Summary: trocr_evaluation_summary.json")

print(f"\n" + "=" * 80)
