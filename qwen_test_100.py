#!/usr/bin/env python3
"""
Quick test: Qwen2.5-VL training on just 100 samples
"""

import os
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from PIL import Image
import numpy as np
import logging

print("=" * 80)
print("🧪 TEST: QWEN2.5-VL ON 100 SAMPLES")
print("=" * 80)

logging.basicConfig(level=logging.ERROR)

# ==================== CONFIG ====================
MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct'
DATASET_NAME = 'shantipriya/odia-ocr-merged'
END_EPOCH = 0.01  # Just ~100 steps
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}\n")

# ==================== LOAD MODEL ====================
print("📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
print(f"✅ Model loaded\n")

# ==================== SETUP LORA ====================
print("⚙️  Configuring LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== LOAD & PREP DATASET ====================
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
print(f"✅ Total available: {len(dataset['train']):,}\n")

# ==================== ROBUST PREPROCESSING (SINGLE PROCESS) ====================
print("🔄 Processing first 100 samples (single process - no hangs)...")

valid_count = 0
failed_count = 0
processed_data = []
max_samples = 100

for idx, example in enumerate(dataset['train']):
    if idx >= max_samples:
        break
    
    try:
        image = example.get('image')
        text = example.get('text', '').strip()
        
        if image is None or not text:
            failed_count += 1
            continue
        
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except:
                failed_count += 1
                continue
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        else:
            failed_count += 1
            continue
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_data.append({'image': image, 'text': text})
        valid_count += 1
        
    except Exception as e:
        failed_count += 1
        continue

print(f"✅ Preprocessing complete!")
print(f"   Valid: {valid_count}")
print(f"   Failed: {failed_count}")
print(f"   Dataset: {len(processed_data)} samples\n")

# ==================== DATA COLLATOR ====================
class QwenCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images, texts = [], []
        
        for example in batch:
            try:
                img = example['image']
                txt = example['text']
                
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                
                images.append(img)
                texts.append(txt)
            except:
                continue
        
        if not images:
            return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}
        
        try:
            inputs = self.processor(
                images, text=texts,
                padding='max_length', truncation=True,
                max_length=2048, return_tensors='pt'
            )
            inputs['labels'] = inputs['input_ids'].clone()
            return inputs
        except Exception as e:
            return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}

# ==================== TRAINING ====================
print("📋 Setting up training (100 steps)...\n")

training_args = TrainingArguments(
    output_dir='./checkpoint-test-100',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    optim='adamw_torch_fused',
    bf16=True,
    report_to=[],
    eval_strategy='no',
    seed=42,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

print("🎯 Creating trainer...\n")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data,
    data_collator=QwenCollator(processor)
)

print("=" * 80)
print("🚀 STARTING TEST TRAINING (100 STEPS)")
print("=" * 80 + "\n")

try:
    trainer.train()
    print("\n" + "=" * 80)
    print("✅ TEST TRAINING COMPLETE!")
    print("=" * 80)
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
