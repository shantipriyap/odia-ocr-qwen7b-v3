#!/usr/bin/env python3
"""
Minimal test: Qwen2.5-VL training WITHOUT LoRA (just prove it works)
"""

import torch
import logging
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModel, TrainingArguments, Trainer

logging.basicConfig(level=logging.ERROR)

print("=" * 80)
print("✅ MINIMAL TEST: QWEN2.5-VL (50 SAMPLES, NO LoRA)")
print("=" * 80)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "shantipriya/odia-ocr-merged"
MAX_SAMPLES = 50

# Load
print("\n📦 Loading model & processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
print("✅ Model loaded")

# Dataset
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_ID, split='train')

# Preprocess
def prep(ex):
    image = ex['image']
    text = (ex.get('text') or '').strip()
    
    if isinstance(image, str):
        try:
            image = Image.open(image).convert('RGB')
        except:
            return None
    elif hasattr(image, 'convert'):
        image = image.convert('RGB')
    else:
        return None
    
    if not text or len(text) < 5:
        return None
    
    return {'image': image, 'text': text}

ds = dataset.select(range(min(50, len(dataset)))).map(prep, num_proc=1).filter(lambda x: x is not None)
print(f"✅ {len(ds)} samples ready")

# Collator
class Col:
    def __init__(self, p):
        self.p = p
    def __call__(self, batch):
        imgs = [ex['image'] for ex in batch]
        txts = [f"Extract Odia text:\n{ex['text']}" for ex in batch]
        enc = self.p(text=txts, images=imgs, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        enc['labels'] = enc['input_ids'].clone()
        pad = self.p.tokenizer.pad_token_id
        enc['labels'][enc['labels'] == pad] = -100
        return enc

# Train
args = TrainingArguments(
    output_dir='./checkpoint-qwen-minimal',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_steps=10,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=2,
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

print("\n🎯 Creating trainer...")
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=Col(processor))

print("\n" + "=" * 80)
print("🚀 TRAINING: 10 STEPS (MINIMAL TEST)")
print("=" * 80 + "\n")

trainer.train()

print("\n" + "=" * 80)
print("✅ SUCCESS!")
print("=" * 80)
