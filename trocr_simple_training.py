#!/usr/bin/env python3
"""
🎯 TrOCR Training - Simple Version WITHOUT Seq2SeqTrainer
Manually handles GPU placement to avoid device issues
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
print("🎯 TROCR TRAINING - SIMPLE APPROACH (No Seq2SeqTrainer)")
print("=" * 80)

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./trocr-odia-simple"
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

# CRITICAL: Fix device placement for position embeddings
print("\n🔧 Fixing device placement...")
model = model.to(DEVICE)

# Explicitly move position embeddings that might be stuck on meta device
if hasattr(model.decoder.model.decoder, 'embed_positions'):
    try:
        model.decoder.model.decoder.embed_positions._float_tensor = \
            model.decoder.model.decoder.embed_positions._float_tensor.to(DEVICE)
        print("✅ Fixed decoder position embeddings")
    except:
        print("⚠️  Could not fix position embeddings (may not exist yet)")

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
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        images = [ex['image'] for ex in batch]
        texts = [ex['text'] for ex in batch]
        
        pixel_values = self.processor(images=images, return_tensors='pt')['pixel_values']
        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': encodings['input_ids'],
            'labels': encodings['input_ids'].clone(),
        }

collator = TrOCRCollator(processor, tokenizer)

# Simple training loop
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

print(f"\n" + "=" * 80)
print(f"🚀 STARTING TRAINING")
print(f"=" * 80 + "\n")

model.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    num_batches = 0
    
    for batch_idx in range(0, len(processed), BATCH_SIZE):
        batch_samples = [processed[i] for i in range(batch_idx, min(batch_idx + BATCH_SIZE, len(processed)))]
        
        try:
            batch = collator(batch_samples)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx // BATCH_SIZE + 1) % 5 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch {epoch+1} | Batch {batch_idx // BATCH_SIZE + 1} | Loss: {avg_loss:.4f}")
        except Exception as e:
            print(f"❌ Error in batch {batch_idx // BATCH_SIZE}: {e}")
            continue

print(f"\n" + "=" * 80)
print(f"✅ TRAINING COMPLETE")
print(f"=" * 80)
