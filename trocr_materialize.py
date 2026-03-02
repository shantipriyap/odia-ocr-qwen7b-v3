#!/usr/bin/env python3
"""
🎯 TrOCR Training - Device Materialization Fix
Load model to CPU first, then move to CUDA to force embedding materialization
"""

import torch
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from torch.optim import AdamW

print("=" * 80)
print("🎯 TROCR TRAINING - Materialization Fix")
print("=" * 80)

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

print(f"\n📦 Device: {DEVICE}")
print(f"🎯 Batch size: {BATCH_SIZE}")

# Load components
print("\n📦 Loading model (CPU first)...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load on CPU first
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, device_map="cpu")
print("✅ Model loaded on CPU")

# Force materialization of all buffers
print("🔧 Materializing all buffers...")
for name, param in model.named_parameters():
    _ = param.device  # Access to check device
for name, buffer in model.named_buffers():
    _ = buffer.device
print("✅ Buffers materialized")

# Now move to CUDA
print(f"🔄 Moving model to {DEVICE}...")
model = model.to(DEVICE)
print(f"✅ Model moved to {DEVICE}")

# Load data
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_data = dataset['train'].select(range(min(100, len(dataset['train']))))  
print(f"✅ Loaded {len(train_data)} samples")

def preprocess(example):
    try:
        image = example['image'].convert('RGB')
        text = example['text'].strip()
        if not text or len(text) < 1:
            return None
        return {'image': image, 'text': text}
    except:
        return None

print("🔄 Preprocessing...")
processed = train_data.map(preprocess, batched=False, num_proc=4)
processed = processed.filter(lambda x: x is not None)
print(f"✅ {len(processed)} samples ready")

class TrOCRCollator:
    def __init__(self, processor, tokenizer, device):
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, batch):
        images = [ex['image'] for ex in batch]
        texts = [ex['text'] for ex in batch]
        
        pixel_values = self.processor(images=images, return_tensors='pt')['pixel_values']
        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        return {
            'pixel_values': pixel_values.to(self.device),
            'decoder_input_ids': encodings['input_ids'].to(self.device),
            'labels': encodings['input_ids'].clone().to(self.device),
        }

collator = TrOCRCollator(processor, tokenizer, DEVICE)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80 + "\n")

model.train()
num_batches = (len(processed) + BATCH_SIZE - 1) // BATCH_SIZE
success_count = 0

for batch_idx in range(0, len(processed), BATCH_SIZE):
    batch_end = min(batch_idx + BATCH_SIZE, len(processed))
    batch = [processed[i] for i in range(batch_idx, batch_end)]
    
    try:
        # Forward pass
        inputs = collator(batch)
        
        outputs = model(
            pixel_values=inputs['pixel_values'],
            decoder_input_ids=inputs['decoder_input_ids'],
            labels=inputs['labels']
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        success_count += 1
        current_batch = batch_idx // BATCH_SIZE + 1
        print(f"✅ Batch {current_batch}/{num_batches} | Loss: {loss.item():.4f}")
        
    except Exception as e:
        current_batch = batch_idx // BATCH_SIZE + 1
        error_msg = str(e).split('\n')[0][:100]
        print(f"❌ Error in batch {current_batch}: {error_msg}")

print("\n" + "=" * 80)
print(f"✅ TRAINING COMPLETE - {success_count} batches succeeded")
print("=" * 80)
