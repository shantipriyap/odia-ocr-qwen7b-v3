#!/usr/bin/env python3
"""
🎯 TrOCR Training - Alternative Model (trocr-base-printed)
Uses different checkpoint without the meta device issue
"""

import torch
import json
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from datetime import datetime
import os

print("=" * 80)
print("🎯 TROCR TRAINING - ALTERNATIVE MODEL (trocr-base-printed)")
print("=" * 80)

# Use printed model checkpoint instead of stage1
MODEL_NAME = "microsoft/trocr-base-printed"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./checkpoint-trocr-printed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n📦 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Device: {DEVICE}")
print(f"   Dataset: {DATASET_NAME}")

# Load components
print("\n📦 Loading model...")
try:
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    print(f"✅ Model loaded: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Error loading {MODEL_NAME}: {e}")
    print("🔄 Falling back to trocr-large-stage1...")
    MODEL_NAME = "microsoft/trocr-large-stage1"
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    print(f"✅ Model loaded: {MODEL_NAME}")

# Add LoRA
print("\n⚙️  Configuring LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none'
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_test_split = dataset['train'].train_test_split(test_size=0.02, seed=42)
train_data = train_test_split['train']
eval_data = train_test_split['test']
print(f"✅ Train: {len(train_data):,} | Eval: {len(eval_data):,} samples")

def preprocess(example):
    try:
        image = example['image'].convert('RGB')
        text = example['text'].strip()
        if not text or len(text) < 1:
            return None
        return {'image': image, 'text': text}
    except:
        return None

print("🔄 Preprocessing training data...")
train_data = train_data.map(preprocess, batched=False, num_proc=8)
train_data = train_data.filter(lambda x: x is not None)
print(f"✅ {len(train_data):,} training samples ready")

print("🔄 Preprocessing eval data...")
eval_data = eval_data.map(preprocess, batched=False, num_proc=8)
eval_data = eval_data.filter(lambda x: x is not None)
print(f"✅ {len(eval_data):,} eval samples ready")

class TrOCRCollator:
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        images = [ex['image'] for ex in batch]
        texts = [ex['text'] for ex in batch]
        
        pixel_values = self.processor(images=images, return_tensors='pt')['pixel_values']
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': encodings['input_ids'],
            'labels': encodings['input_ids'].clone(),
        }

collator = TrOCRCollator(processor, tokenizer)

# Training args
print("\n📋 Setting up training...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_steps=300,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,
    predict_with_generate=False,
    remove_unused_columns=False,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    bf16=torch.cuda.is_available(),
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    seed=42,
    report_to=[],
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

print("✅ Training arguments configured")

# Create trainer
print("\n🎯 Creating trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=collator,
)
print("✅ Trainer ready")

# Start training
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)

try:
    train_result = trainer.train()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    print(f"   Duration: {train_result.training_steps} steps")
    
    # Save final model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Model saved!")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
