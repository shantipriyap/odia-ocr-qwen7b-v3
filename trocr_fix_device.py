#!/usr/bin/env python3
"""
🎯 TrOCR Training - FIXED Device Handling
Move to CUDA BEFORE applying LoRA to avoid meta device issues
"""

import torch
import json
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from datetime import datetime

print("=" * 80)
print("🎯 TROCR TRAINING - FIX DEVICE (Move to CUDA BEFORE LoRA)")
print("=" * 80)

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./trocr-odia-fixed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 1  # Just 1 epoch for testing
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

print(f"\n📦 Device: {DEVICE}")
print(f"🎯 Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

# Load components
print("\n📦 Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# ⭐ CRITICAL FIX: Move to CUDA FIRST, then apply LoRA
print("🔄 Moving model to CUDA BEFORE applying LoRA...")
model = model.to(DEVICE)
print("✅ Model moved to CUDA")

# Now apply LoRA while model is already on CUDA
print("⚙️  Adding LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none'
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load data
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_data = dataset['train'].select(range(min(100, len(dataset['train']))))  # 100 samples for quick test
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
gradient_accumulation_steps = 1
num_batches = (len(processed) + BATCH_SIZE - 1) // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    
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
            
            total_loss += loss.item()
            
            current_batch = batch_idx // BATCH_SIZE + 1
            print(f"✅ Batch {current_batch}/{num_batches} | Loss: {loss.item():.4f}")
            
        except Exception as e:
            current_batch = batch_idx // BATCH_SIZE + 1
            print(f"❌ Error in batch {current_batch}: {str(e)[:80]}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"\n📊 Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}\n")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)

# Save model
print(f"\n💾 Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
print("✅ Model saved!")
