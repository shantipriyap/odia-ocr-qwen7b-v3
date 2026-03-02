#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE OCR MODEL BENCHMARKING SUITE
=============================================
Compare Qwen2.5-VL against other OCR models on Odia text

Models tested:
  1. Qwen2.5-VL-3B-Instruct (fine-tuned) - YOUR MODEL
  2. PaddleOCR (multilingual baseline)
  3. TrOCR (Vision Transformer)
  4. EasyOCR (fast multilingual)
  5. Qwen2.5-VL (base model - no fine-tuning)

Metrics:
  • CER (Character Error Rate) - LOWER is better
  • WER (Word Error Rate)
  • BLEU Score
  • Inference Time
  • Memory Usage
  • Throughput (images/sec)
"""

import sys
import os
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from datasets import load_dataset
import torch
import concurrent.futures
from tqdm import tqdm

print("=" * 80)
print("🔍 OCR MODEL BENCHMARKING SUITE - INITIALIZATION")
print("=" * 80)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    # Simple Levenshtein-like calculation
    edits = 0
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    # Use dynamic programming for edit distance
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j
    
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[ref_len][hyp_len] / ref_len


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Edit distance for words
    edits = 0
    ref_len = len(ref_words)
    hyp_len = len(hyp_words)
    
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j
    
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[ref_len][hyp_len] / ref_len


def calculate_bleu(reference: str, hypothesis: str, n: int = 4) -> float:
    """Calculate BLEU score (simplified)"""
    from collections import Counter
    
    # Split into n-grams
    ref_ngrams = []
    hyp_ngrams = []
    
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    for i in range(len(ref_chars) - n + 1):
        ref_ngrams.append(tuple(ref_chars[i:i+n]))
    
    for i in range(len(hyp_chars) - n + 1):
        hyp_ngrams.append(tuple(hyp_chars[i:i+n]))
    
    if not hyp_ngrams:
        return 0.0
    
    ref_counts = Counter(ref_ngrams)
    hyp_counts = Counter(hyp_ngrams)
    
    matches = sum((ref_counts & hyp_counts).values())
    return matches / len(hyp_ngrams) if hyp_ngrams else 0.0


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

class QwenOCRModel:
    """Qwen2.5-VL OCR Model"""
    
    def __init__(self, model_id: str, use_fine_tuned: bool = True):
        self.name = "Qwen2.5-VL" + (" (Fine-tuned)" if use_fine_tuned else "")
        self.model_id = model_id
        self.use_fine_tuned = use_fine_tuned
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            from transformers import AutoProcessor, AutoModel
            
            print(f"  📦 Loading {self.name}...")
            
            # Load processor from base model
            processor_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            self.processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
            
            # Load model
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            print(f"  ❌ Error loading {self.name}: {e}")
            self.model = None
            self.processor = None
    
    def infer_batch(self, images: List[Image.Image]) -> List[Tuple[str, float, float]]:
        """Batch inference for multiple images"""
        results = []
        if self.model is None or self.processor is None:
            return [("", 0.0, 0.0)] * len(images)
        try:
            # Resize images if needed
            max_size = 1024
            batch = []
            for img in images:
                if img.width > max_size or img.height > max_size:
                    img = img.copy()
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                batch.append(img)
            start_time = time.time()
            tracemalloc.start()
            with torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device, self.dtype)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False
                )
                generated_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
            elapsed_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024 / 1024
            for text in generated_texts:
                results.append((text, elapsed_time / len(batch), memory_mb / len(batch)))
            return results
        except Exception as e:
            print(f"    ⚠️  Batch inference error: {str(e)[:100]}")
            return [("", 0.0, 0.0)] * len(images)


class PaddleOCRModel:
    """PaddleOCR Model"""
    
    def __init__(self):
        self.name = "PaddleOCR"
        try:
            from paddleocr import PaddleOCR
            print(f"  📦 Loading {self.name}...")
            self.model = PaddleOCR(use_angle_cls=True, lang='ch')  # CJK includes some Indic
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            print(f"  ❌ Error loading {self.name}: {e}")
            self.model = None
    
    def infer(self, image: Image.Image) -> Tuple[str, float, float]:
        """Infer text from image"""
        if self.model is None:
            return "", 0.0, 0.0
        
        try:
            import numpy as np
            
            start_time = time.time()
            tracemalloc.start()
            
            # Convert PIL to array
            img_array = np.array(image)
            
            # Inference
            result = self.model.ocr(img_array, cls=True)
            
            elapsed_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024 / 1024
            
            # Extract text
            text = ""
            if result:
                for line in result:
                    if line:
                        for word_info in line:
                            text += word_info[1] + " "
            
            return text.strip(), elapsed_time, memory_mb
        except Exception as e:
            print(f"    ⚠️  Inference error: {e}")
            return "", 0.0, 0.0


class EasyOCRModel:
    """EasyOCR Model"""
    
    def __init__(self):
        self.name = "EasyOCR"
        try:
            import easyocr
            print(f"  📦 Loading {self.name}...")
            self.model = easyocr.Reader(['en', 'ch'])  # English + Chinese as proxy
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            print(f"  ❌ Error loading {self.name}: {e}")
            self.model = None
    
    def infer(self, image: Image.Image) -> Tuple[str, float, float]:
        """Infer text from image"""
        if self.model is None:
            return "", 0.0, 0.0
        
        try:
            import numpy as np
            
            start_time = time.time()
            tracemalloc.start()
            
            # Convert PIL to array
            img_array = np.array(image)
            
            # Inference
            results = self.model.readtext(img_array)
            
            elapsed_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024 / 1024
            
            # Extract text
            text = " ".join([result[1] for result in results if result])
            
            return text, elapsed_time, memory_mb
        except Exception as e:
            print(f"    ⚠️  Inference error: {e}")
            return "", 0.0, 0.0


class TrOCRModel:
    """TrOCR Model"""
    
    def __init__(self):
        self.name = "TrOCR"
        try:
            from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
            
            print(f"  📦 Loading {self.name}...")
            model_name = "microsoft/trocr-base-handwritten"
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.model.eval()
            self.device = device
            
            print(f"  ✅ {self.name} loaded")
        except Exception as e:
            print(f"  ❌ Error loading {self.name}: {e}")
            self.model = None
            self.feature_extractor = None
            self.tokenizer = None
    
    def infer_batch(self, images: List[Image.Image]) -> List[Tuple[str, float, float]]:
        """Batch inference for multiple images"""
        results = []
        if self.model is None:
            return [("", 0.0, 0.0)] * len(images)
        try:
            start_time = time.time()
            tracemalloc.start()
            pixel_values = self.feature_extractor(images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=128)
                texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            elapsed_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024 / 1024
            for text in texts:
                results.append((text, elapsed_time / len(images), memory_mb / len(images)))
            return results
        except Exception as e:
            print(f"    ⚠️  Batch inference error: {e}")
            return [("", 0.0, 0.0)] * len(images)


# ============================================================================
# MAIN BENCHMARKING
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("📊 LOADING TEST DATASET")
    print("=" * 80)
    
    try:
        print("  📥 Starting dataset load...")
        dataset = load_dataset("shantipriya/odia-ocr-merged", split="train")
        sample_size = min(20, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        def load_and_convert(idx):
            sample = dataset[int(idx)]
            if "image" in sample and "text" in sample:
                image = sample["image"]
                if isinstance(image, str):
                    try:
                        image = Image.open(image).convert("RGB")
                    except:
                        return None
                elif not isinstance(image, Image.Image):
                    try:
                        image = Image.fromarray(image).convert("RGB")
                    except:
                        return None
                return {"image": image, "text": sample["text"]}
            return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            test_samples = list(filter(None, executor.map(load_and_convert, indices)))
        if not test_samples:
            print("  ❌ No valid samples found")
            return
        print(f"  ✅ {len(test_samples)} samples ready")
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        print("\n  📝 Using synthetic samples instead...")
        test_samples = []
        for i in range(5):
            img = Image.new('RGB', (200, 100), color='white')
            test_samples.append({"image": img, "text": f"Odia text sample {i+1}"})
    
    # ========================================================================
    # INITIALIZE MODELS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("🤖 INITIALIZING MODELS")
    print("=" * 80)
    
    models = {}
    
    # Qwen Fine-tuned
    try:
        models["Qwen2.5-VL (Fine-tuned)"] = QwenOCRModel(
            "shantipriya/odia-ocr-qwen-finetuned",
            use_fine_tuned=True
        )
    except:
        print("  ⚠️  Qwen fine-tuned model not available")
    
    # Qwen Base
    try:
        models["Qwen2.5-VL (Base)"] = QwenOCRModel(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            use_fine_tuned=False
        )
    except:
        print("  ⚠️  Qwen base model not available")
    
    # PaddleOCR
    try:
        models["PaddleOCR"] = PaddleOCRModel()
    except:
        print("  ⚠️  PaddleOCR not available")
    
    # EasyOCR
    try:
        models["EasyOCR"] = EasyOCRModel()
    except:
        print("  ⚠️  EasyOCR not available")
    
    # TrOCR
    try:
        models["TrOCR"] = TrOCRModel()
    except:
        print("  ⚠️  TrOCR not available")
    
    if not models:
        print("  ❌ No models could be loaded")
        return
    
    print(f"\n  ✅ {len(models)} models ready for benchmarking")
    
    # ========================================================================
    # RUN BENCHMARKS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("🏃 RUNNING BENCHMARKS")
    print("=" * 80)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Testing {model_name}...")
        print(f"  {'─' * 76}")
        cer_scores = []
        wer_scores = []
        bleu_scores = []
        times = []
        memory_usage = []
        batch_size = 4 if hasattr(model, 'infer_batch') else 1
        for batch_start in tqdm(range(0, len(test_samples), batch_size), desc=f"{model_name} benchmarking"):
            batch_samples = test_samples[batch_start:batch_start+batch_size]
            images = [s["image"] for s in batch_samples]
            refs = [s["text"] for s in batch_samples]
            if hasattr(model, 'infer_batch'):
                batch_results = model.infer_batch(images)
            else:
                batch_results = [model.infer(img) if hasattr(model, 'infer') else ("", 0.0, 0.0) for img in images]
            for i, (hypothesis_text, elapsed, mem) in enumerate(batch_results):
                hypothesis_text = hypothesis_text.strip() if hypothesis_text else ""
                reference_text = refs[i].strip() if refs[i] else ""
                cer = calculate_cer(reference_text, hypothesis_text) if reference_text else 0.0
                wer = calculate_wer(reference_text, hypothesis_text) if reference_text else 0.0
                bleu = calculate_bleu(reference_text, hypothesis_text) if reference_text else 0.0
                cer_scores.append(cer)
                wer_scores.append(wer)
                bleu_scores.append(bleu)
                times.append(elapsed)
                memory_usage.append(mem)
        avg_cer = np.mean(cer_scores) if cer_scores else 0.0
        avg_wer = np.mean(wer_scores) if wer_scores else 0.0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_time = np.mean(times) if times else 0.0
        avg_mem = np.mean(memory_usage) if memory_usage else 0.0
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        results[model_name] = {
            "cer": float(avg_cer),
            "cer_std": float(np.std(cer_scores)) if cer_scores else 0.0,
            "wer": float(avg_wer),
            "wer_std": float(np.std(wer_scores)) if wer_scores else 0.0,
            "bleu": float(avg_bleu),
            "bleu_std": float(np.std(bleu_scores)) if bleu_scores else 0.0,
            "avg_time_s": float(avg_time),
            "time_std": float(np.std(times)) if times else 0.0,
            "memory_mb": float(avg_mem),
            "throughput_img_per_sec": float(throughput),
            "samples": len(test_samples)
        }
        print(f"  ✅ Complete: CER={avg_cer:.1%}, Time={avg_time:.2f}s, Mem={avg_mem:.1f}MB")
    
    # ========================================================================
    # GENERATE REPORT
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by CER (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["cer"])
    
    print("\n🥇 RANKINGS BY ACCURACY (Character Error Rate - Lower is Better)")
    print("─" * 80)
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"  {rank}."
        print(f"{medal} {name:30} | CER: {metrics['cer']:7.2%} ± {metrics['cer_std']:6.2%} | "
              f"Speed: {metrics['avg_time_s']:5.2f}s | Mem: {metrics['memory_mb']:7.1f}MB")
    
    # Sort by speed
    sorted_by_speed = sorted(results.items(), key=lambda x: x[1]["avg_time_s"])
    
    print("\n⚡ RANKINGS BY SPEED (Lower Time is Better)")
    print("─" * 80)
    for rank, (name, metrics) in enumerate(sorted_by_speed[:5], 1):
        medal = "🏃" if rank == 1 else f"  {rank}."
        print(f"{medal} {name:30} | Speed: {metrics['avg_time_s']:6.2f}s | "
              f"Throughput: {metrics['throughput_img_per_sec']:6.2f} img/s | "
              f"CER: {metrics['cer']:7.2%}")
    
    # Sort by memory
    sorted_by_mem = sorted(results.items(), key=lambda x: x[1]["memory_mb"])
    
    print("\n💾 RANKINGS BY MEMORY EFFICIENCY (Lower is Better)")
    print("─" * 80)
    for rank, (name, metrics) in enumerate(sorted_by_mem[:5], 1):
        medal = "✨" if rank == 1 else f"  {rank}."
        print(f"{medal} {name:30} | Memory: {metrics['memory_mb']:7.1f}MB | "
              f"Speed: {metrics['avg_time_s']:5.2f}s | CER: {metrics['cer']:7.2%}")
    
    # ========================================================================
    # DETAILED METRICS TABLE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📊 DETAILED METRICS TABLE")
    print("=" * 80)
    
    print(f"\n{'Model':30} | {'CER':>8} | {'WER':>8} | {'BLEU':>8} | {'Time':>7} | {'Mem':>7} | {'TPM':>8}")
    print("─" * 80)
    
    for name, metrics in sorted_results:
        print(f"{name:30} | {metrics['cer']:>7.2%} | {metrics['wer']:>7.2%} | "
              f"{metrics['bleu']:>7.2%} | {metrics['avg_time_s']:>6.2f}s | "
              f"{metrics['memory_mb']:>6.1f}MB | {metrics['throughput_img_per_sec']:>7.3f}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_file = "/Users/shantipriya/work/odia_ocr/benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    best_accuracy = sorted_results[0]
    best_speed = sorted_by_speed[0]
    best_memory = sorted_by_mem[0]
    
    print(f"\n🎯 Your Model: Qwen2.5-VL (Fine-tuned)")
    if "Qwen2.5-VL (Fine-tuned)" in results:
        your_metrics = results["Qwen2.5-VL (Fine-tuned)"]
        print(f"   • Accuracy (CER): {your_metrics['cer']:7.2%} {('✅ Best!' if best_accuracy[0] == 'Qwen2.5-VL (Fine-tuned)' else '')}")
        print(f"   • Speed: {your_metrics['avg_time_s']:6.2f}s")
        print(f"   • Memory: {your_metrics['memory_mb']:7.1f}MB")
        print(f"   • Throughput: {your_metrics['throughput_img_per_sec']:6.2f} img/s")
    
    print(f"\n🏆 Best Accuracy: {best_accuracy[0]}")
    print(f"   CER: {best_accuracy[1]['cer']:.2%}")
    
    print(f"\n⚡ Fastest: {best_speed[0]}")
    print(f"   Time: {best_speed[1]['avg_time_s']:.2f}s ({best_speed[1]['throughput_img_per_sec']:.2f} img/s)")
    
    print(f"\n💾 Most Memory Efficient: {best_memory[0]}")
    print(f"   Memory: {best_memory[1]['memory_mb']:.1f}MB")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
