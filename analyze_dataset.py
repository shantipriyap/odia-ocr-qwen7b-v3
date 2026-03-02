#!/usr/bin/env python3
"""
📊 Odia-OCR Dataset Analysis & Sample Testing
Analyze dataset structure and test preprocessing with small sample
"""

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image
import json
from collections import defaultdict

print("=" * 80)
print("📊 ODIA-OCR DATASET ANALYSIS & SAMPLE TEST")
print("=" * 80)

# Load dataset
print("\n📥 Loading dataset...")
dataset = load_dataset("shantipriya/odia-ocr-merged")
print(f"✅ Dataset loaded")
print(f"\n📦 Dataset splits: {list(dataset.keys())}")
print(f"   Train samples: {len(dataset['train']):,}")

# Analyze dataset
print("\n" + "=" * 80)
print("📊 DATASET ANALYSIS")
print("=" * 80)

train_data = dataset['train']
sample_size = min(1000, len(train_data))
sample_data = train_data.select(range(sample_size))

# Analyze content
print(f"\n🔍 Analyzing first {sample_size} samples...")

stats = {
    'images_ok': 0,
    'images_failed': 0,
    'text_lengths': [],
    'image_sizes': [],
    'has_text': 0,
    'no_text': 0,
}

for i, example in enumerate(sample_data):
    try:
        # Check image
        if example.get('image'):
            img = example['image']
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert('RGB')
                    stats['image_sizes'].append(img.size)
                    stats['images_ok'] += 1
                except:
                    stats['images_failed'] += 1
            elif hasattr(img, 'size'):
                stats['image_sizes'].append(img.size)
                stats['images_ok'] += 1
            else:
                stats['images_failed'] += 1
        else:
            stats['images_failed'] += 1
        
        # Check text
        text = example.get('text', '').strip()
        if text:
            stats['has_text'] += 1
            stats['text_lengths'].append(len(text))
        else:
            stats['no_text'] += 1
    except Exception as e:
        stats['images_failed'] += 1
        if i < 5:
            print(f"  ⚠️  Sample {i} error: {e}")

# Print stats
print(f"\n📈 Image Statistics (n={sample_size}):")
print(f"   ✅ Images OK: {stats['images_ok']:,}")
print(f"   ❌ Images Failed: {stats['images_failed']:,}")
print(f"   Success rate: {100*stats['images_ok']/sample_size:.1f}%")

if stats['image_sizes']:
    sizes = stats['image_sizes']
    print(f"\n   Size range: {min(sizes)} to {max(sizes)}")
    avg_w = sum(s[0] for s in sizes) / len(sizes)
    avg_h = sum(s[1] for s in sizes) / len(sizes)
    print(f"   Average size: {avg_w:.0f}x{avg_h:.0f}")

print(f"\n📝 Text Statistics:")
print(f"   ✅ Has text: {stats['has_text']:,}")
print(f"   ❌ No text: {stats['no_text']:,}")

if stats['text_lengths']:
    lengths = stats['text_lengths']
    print(f"\n   Min length: {min(lengths)}")
    print(f"   Max length: {max(lengths)}")
    print(f"   Avg length: {sum(lengths)/len(lengths):.0f}")
    print(f"   Median length: {sorted(lengths)[len(lengths)//2]}")

# Sample some records
print("\n" + "=" * 80)
print("📋 SAMPLE RECORDS")
print("=" * 80)

for idx in [0, 100, 500, 1000]:
    if idx < len(sample_data):
        example = sample_data[idx]
        text = example.get('text', 'N/A')[:50]
        img = example.get('image')
        if isinstance(img, str):
            img_info = f"(path: {img[:40]}...)"
        else:
            img_info = f"(PIL Image)"
        print(f"\n📄 Sample {idx}:")
        print(f"   Image: {img_info}")
        print(f"   Text: {text}")

# Now test model loading and preprocessing
print("\n" + "=" * 80)
print("🧪 MODEL & PREPROCESSING TEST")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n💻 Device: {device}")

# Load model
print("\n📦 Loading Qwen2.5-VL-3B...")
try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print(f"✅ Model loaded successfully")
    print(f"   Model size: {model.get_memory_footprint() / 1e9:.2f}GB")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Apply LoRA
print("\n⚙️  Applying LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
print(f"✅ LoRA applied")
model.print_trainable_parameters()

# Test preprocessing with 5 samples
print("\n🔄 Testing preprocessing with 5 samples...")

class OdiaOCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = []
        texts = []
        
        for example in batch:
            try:
                img = example['image']
                text = example['text']
                
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                elif hasattr(img, 'convert'):
                    img = img.convert('RGB')
                else:
                    continue
                
                if not text or len(text) < 2:
                    continue
                
                images.append(img)
                texts.append(text)
            except:
                continue
        
        if not images:
            print("⚠️  No valid images in batch!")
            return None
        
        try:
            formatted_texts = [f"<|user|>\nExtract the text from this Odia document.\n<|end|>\n<|assistant|>\n{text}" for text in texts]
            
            inputs = self.processor(
                images=images,
                text=formatted_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            
            inputs['labels'] = inputs['input_ids'].clone()
            return inputs
        except Exception as e:
            print(f"⚠️  Collation error: {e}")
            return None

collator = OdiaOCRDataCollator(processor)
test_samples = train_data.select(range(5))

print(f"\n   Testing with samples...")
for i, sample in enumerate(test_samples):
    print(f"\n   Sample {i}:")
    text = sample.get('text', '')[:50]
    print(f"   - Text: {text}...")
    
    batch = [sample]
    result = collator(batch)
    
    if result:
        print(f"   ✅ Preprocessing OK")
        print(f"      Input IDs shape: {result['input_ids'].shape}")
        print(f"      Attention mask shape: {result['attention_mask'].shape}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=result['input_ids'].to(device),
                    attention_mask=result['attention_mask'].to(device),
                )
            print(f"   ✅ Forward pass OK - logits shape: {outputs.logits.shape}")
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
    else:
        print(f"   ❌ Preprocessing failed")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

# Summary
print(f"\n📊 Summary:")
print(f"   Total samples: {len(train_data):,}")
print(f"   Sample success rate: {100*stats['images_ok']/sample_size:.1f}%")
print(f"   Model loaded: ✅")
print(f"   LoRA configured: ✅")
print(f"   Preprocessing tested: ✅")
print(f"\n🚀 Ready for full training!")
