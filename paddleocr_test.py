#!/usr/bin/env python3
"""
PaddleOCR Test on Odia Dataset
Test PaddleOCR baseline on your merged Odia dataset
"""

import os
import sys
from pathlib import Path
import tempfile
import time

print("=" * 80)
print("🧪 PADDLEOCR BASELINE TEST - ODIA OCR DATASET")
print("=" * 80)

# Install PaddleOCR if needed
print("\n📦 Installing PaddleOCR...")
os.system("pip install -q paddleocr pillow datasets 2>/dev/null")

print("✅ Dependencies installed")

try:
    from paddleocr import PaddleOCR
    from datasets import load_dataset
    from PIL import Image as PILImage
    import difflib
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Character Error Rate calculator
def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    total = len(reference)
    if total == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return 1.0 - (matches / total)

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    total = len(ref_words)
    if total == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return 1.0 - (matches / total)

# Initialize PaddleOCR
print("\n🚀 Initializing PaddleOCR...")
try:
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',  # Use English model (works for Odia too)
        show_log=False
    )
    print("✅ PaddleOCR initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

# Load dataset
print("\n📥 Loading dataset...")
try:
    dataset = load_dataset("shantipriya/odia-ocr-merged", split="train")
    print(f"✅ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Test on samples
print("\n" + "=" * 80)
print("🔍 TESTING ON SAMPLES")
print("=" * 80)

test_sizes = [5, 10, 20]
results = {}

for num_samples in test_sizes:
    print(f"\n📊 Testing on {num_samples} random samples...")
    print("-" * 80)
    
    samples = dataset.select(range(min(num_samples, len(dataset))))
    
    total_cer = 0
    total_wer = 0
    total_exact_match = 0
    successful = 0
    
    start_time = time.time()
    
    for idx, item in enumerate(samples):
        try:
            # Get image and text
            img = item['image']
            true_text = item['text'].strip()
            
            # Convert image if needed
            if isinstance(img, str):
                img = PILImage.open(img).convert('RGB')
            else:
                img = img.convert('RGB')
            
            # Save temp image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                temp_path = tmp.name
            
            # Run OCR
            ocr_result = ocr.ocr(temp_path, cls=True)
            
            # Extract text
            if ocr_result and ocr_result[0]:
                ocr_text = ' '.join([line[1][0] for line in ocr_result[0]]).strip()
            else:
                ocr_text = ""
            
            # Calculate metrics
            cer = calculate_cer(true_text, ocr_text)
            wer = calculate_wer(true_text, ocr_text)
            exact_match = 1 if true_text.lower() == ocr_text.lower() else 0
            
            total_cer += cer
            total_wer += wer
            total_exact_match += exact_match
            successful += 1
            
            # Show sample results
            if idx < 3:  # Show first 3
                print(f"\n  Sample {idx + 1}:")
                print(f"    True:  {true_text[:70]}")
                print(f"    OCR:   {ocr_text[:70]}")
                print(f"    CER:   {cer*100:.1f}%  |  WER: {wer*100:.1f}%")
            
            # Cleanup
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"  ❌ Error on sample {idx}: {e}")
    
    elapsed = time.time() - start_time
    
    # Calculate averages
    if successful > 0:
        avg_cer = (total_cer / successful) * 100
        avg_wer = (total_wer / successful) * 100
        accuracy = (total_exact_match / successful) * 100
        
        results[num_samples] = {
            'avg_cer': avg_cer,
            'avg_wer': avg_wer,
            'accuracy': accuracy,
            'successful': successful,
            'time': elapsed
        }
        
        print(f"\n  📈 Results for {num_samples} samples:")
        print(f"     Avg CER: {avg_cer:.2f}%")
        print(f"     Avg WER: {avg_wer:.2f}%")
        print(f"     Exact Match Accuracy: {accuracy:.2f}%")
        print(f"     Processed: {successful}/{num_samples} ({100*successful/num_samples:.0f}%)")
        print(f"     Time: {elapsed:.1f}s ({elapsed/successful:.2f}s per sample)")

# Print summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

for num_samples, metrics in results.items():
    print(f"\n{num_samples} Samples:")
    print(f"  Character Error Rate (CER):    {metrics['avg_cer']:.2f}%")
    print(f"  Word Error Rate (WER):         {metrics['avg_wer']:.2f}%")
    print(f"  Exact Match Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"  Processing Speed:              {metrics['time']/metrics['successful']:.2f}s/sample")

# Recommendations
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

if results:
    best_accuracy = max(r['accuracy'] for r in results.values())
    if best_accuracy > 90:
        print("\n✅ GOOD: PaddleOCR shows >90% accuracy!")
        print("   → Use as production baseline")
        print("   → No fine-tuning needed")
    elif best_accuracy > 70:
        print("\n⚠️  MODERATE: PaddleOCR shows 70-90% accuracy")
        print("   → Could benefit from fine-tuning")
        print("   → Consider TrOCR for better results")
    else:
        print("\n❌ POOR: PaddleOCR shows <70% accuracy")
        print("   → Fine-tuning recommended")
        print("   → Use TrOCR or other specialized model")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)
