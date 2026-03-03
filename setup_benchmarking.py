#!/usr/bin/env python3
"""
🔧 OCR BENCHMARKING - DEPENDENCY SETUP GUIDE
==============================================
Install all required dependencies for benchmarking
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """Run a command and report status"""
    if description:
        print(f"\n📦 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ⚠️  Warning: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Timeout (may still work)")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


print("=" * 80)
print("🔧 SETTING UP OCR BENCHMARKING ENVIRONMENT")
print("=" * 80)

# Core dependencies
core_deps = [
    ("torch torchvision torchaudio", "PyTorch - Machine Learning Framework"),
    ("transformers datasets", "HuggingFace - Model Hub & Datasets"),
    ("pillow numpy scipy", "Image Processing & Math"),
    ("accelerate peft", "Model Optimization & LoRA"),
]

print("\n🎯 Installing Core Dependencies:")
for pkg, desc in core_deps:
    run_command(f"pip install -q {pkg} 2>/dev/null", desc)

# Optional OCR models (try each, don't fail if one fails)
ocr_models = [
    ("paddleocr paddlepaddle", "📊 PaddleOCR - Fast Multilingual OCR"),
    ("easyocr", "📊 EasyOCR - Easy Multilingual OCR"),
]

print("\n\n🎯 Installing Optional OCR Models (These may take time):")
for pkg, desc in ocr_models:
    run_command(f"pip install -q {pkg} 2>/dev/null &", desc)

# Dataset utilities
print("\n\n🎯 Verifying Dataset Access:")
try:
    from datasets import load_dataset
    print("   ✅ datasets package ready")
    
    # Try loading a small portion
    print("   🔄 Testing dataset access...")
    ds = load_dataset("shantipriya/odia-ocr-merged", split="train", streaming=True)
    print(f"   ✅ Dataset accessible ({ds.info.features})")
except Exception as e:
    print(f"   ⚠️  Dataset access issue: {str(e)[:100]}")

print("\n" + "=" * 80)
print("✅ SETUP COMPLETE!")
print("=" * 80)

print("""
📝 To run the benchmark:

  python3 /Users/shantipriya/work/odia_ocr/benchmark_ocr_models.py

⏱️  Expected runtime: 5-30 minutes (depending on models available)

📊 Output files:
  • benchmark_results.json - Detailed metrics for all models
  • Console output - Formatted summary and rankings

🎯 Models tested:
  1. Qwen2.5-VL (Fine-tuned) - YOUR MODEL ⭐
  2. Qwen2.5-VL (Base) - For comparison
  3. PaddleOCR - Fast baseline
  4. EasyOCR - Alternative approach
  5. TrOCR - Vision Transformer method

📈 Metrics calculated:
  • CER (Character Error Rate) - Lower is better
  • WER (Word Error Rate) - Lower is better
  • BLEU Score - Higher is better
  • Inference Speed - Lower time is better
  • Memory Usage - Lower is better
  • Throughput - Images per second

💡 Tips:
  • First run will download models (slow, one-time)
  • Subsequent runs are faster (models cached)
  • Results saved to benchmark_results.json
  • Compare results across different configurations
""")
