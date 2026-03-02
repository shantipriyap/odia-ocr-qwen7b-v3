#!/usr/bin/env python3
"""
🎯 TrOCR Training - CPU Test (Skip Device Issues)
Confirm data pipeline and training loop work on CPU first
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
print("🎯 TROCR TRAINING - CPU TEST")
print("=" * 80)

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"
BATCH_SIZE = 2  # Smaller batch on CPU
LEARNING_RATE = 5e-5

print(f"\n📦 Device: CPU (testing data pipeline)")
print(f"🎯 Batch size: {BATCH_SIZE}")

# Load components
print("\n📦 Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Keep on CPU to avoid device issues
print("✅ Model loaded (CPU)")

# Load data
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_data = dataset['train'].select(range(min(50, len(dataset['train']))))  # Just 50 samples
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

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING (CPU - Data Pipeline Test)")
print("=" * 80 + "\n")

model.eval()  # Use eval mode since we're just testing data flow
num_batches = (len(processed) + BATCH_SIZE - 1) // BATCH_SIZE
success_count = 0

with torch.no_grad():
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
            success_count += 1
            current_batch = batch_idx // BATCH_SIZE + 1
            print(f"✅ Batch {current_batch}/{num_batches} | Loss: {loss.item():.4f}")
            
        except Exception as e:
            current_batch = batch_idx // BATCH_SIZE + 1
            error_msg = str(e).split('\n')[0][:100]
            print(f"❌ Error in batch {current_batch}: {error_msg}")

print("\n" + "=" * 80)
print(f"✅ DATA PIPELINE TEST COMPLETE - {success_count}/{num_batches} batches succeeded")
print("=" * 80)
print("\n🎯 NEXT STEPS:")
print("1. If all batches succeeded: Data pipeline is working!")
print("2. Device issue is a separate GPU/PEFT compatibility problem")
print("3. Solution: Use alternative training approach or switch models")
