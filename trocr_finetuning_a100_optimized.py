#!/usr/bin/env python3
"""
🎯 TrOCR Fine-tuning for Odia OCR - OPTIMIZED FOR A100
Maximum accuracy approach using Vision Encoder-Decoder
"""

import torch
import json
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model
import os
from PIL import Image
import numpy as np
from datetime import datetime

print("=" * 80)
print("🚀 TrOCR FINE-TUNING FOR ODIA OCR ON A100")
print("=" * 80)

# Check GPU
print(f"\n✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# Configuration
MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./trocr-odia-finetuned"
CHECKPOINT_DIR = "./trocr-checkpoints"

# Hyperparameters optimized for A100
config = {
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 2,
    'num_train_epochs': 3,
    'learning_rate': 5e-5,
    'warmup_steps': 500,
    'max_steps': -1,
    'weight_decay': 0.01,
    'eval_strategy': 'steps',
    'eval_steps': 500,
    'save_steps': 500,
    'save_total_limit': 3,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'cer',
    'greater_is_better': False,
}

print(f"\n📋 Configuration:")
for key, val in config.items():
    print(f"   {key}: {val}")

# Load dataset
print(f"\n📥 Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)
train_data = dataset['train']
print(f"✅ Loaded {len(train_data)} samples")

# Load model components
print(f"\n📦 Loading model: {MODEL_NAME}")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Set model config for Odia script
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

print(f"✅ Model loaded with vocab size: {model.config.vocab_size}")

# Apply LoRA for efficient fine-tuning
print(f"\n⚙️ Applying LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=['q_proj', 'v_proj', 'k_proj'],
    lora_dropout=0.1,
    bias='none',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data preprocessing
print(f"\n🔄 Preprocessing dataset...")

def preprocess_images(batch):
    """Preprocess images and text"""
    images, texts = [], []
    
    for i in range(len(batch['image'])):
        try:
            image = batch['image'][i]
            text = batch['text'][i].strip()
            
            if not text or len(text) < 1:
                continue
            
            # Convert PIL to numpy if needed
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            images.append(image)
            texts.append(text)
        except:
            continue
    
    if not images:
        return {'pixel_values': [], 'decoder_input_ids': [], 'labels': []}
    
    # Process images
    try:
        processed = processor(images=images, return_tensors='pt', padding=True)
    except:
        return {'pixel_values': [], 'decoder_input_ids': [], 'labels': []}
    
    # Tokenize text
    try:
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
    except:
        return {'pixel_values': [], 'decoder_input_ids': [], 'labels': []}
    
    return {
        'pixel_values': processed['pixel_values'],
        'decoder_input_ids': encodings['input_ids'],
        'labels': encodings['input_ids'].clone(),
    }

# Preprocess and filter
processed_dataset = train_data.map(
    preprocess_images,
    batched=True,
    batch_size=32,
    remove_columns=['image', 'text'],
    num_proc=4
)

# Filter out empty samples
processed_dataset = processed_dataset.filter(
    lambda x: len(x['pixel_values']) > 0 if isinstance(x['pixel_values'], list) else x['pixel_values'].shape[0] > 0
)

print(f"✅ Processed {len(processed_dataset)} valid samples")

# Split for evaluation
if len(processed_dataset) > 10000:
    split_data = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_data['train']
    eval_dataset = split_data['test']
else:
    train_dataset = processed_dataset
    eval_dataset = processed_dataset.select(range(min(100, len(processed_dataset))))

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Initialize trainer
print(f"\n🎯 Initializing Seq2SeqTrainer...")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    warmup_steps=config['warmup_steps'],
    weight_decay=config['weight_decay'],
    eval_strategy=config['eval_strategy'],
    eval_steps=config['eval_steps'],
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    load_best_model_at_end=config['load_best_model_at_end'],
    metric_for_best_model=config['metric_for_best_model'],
    greater_is_better=config['greater_is_better'],
    logging_steps=50,
    report_to=[],
    save_strategy="steps",
    bf16=True,
    dataloader_num_workers=4,
    predict_with_generate=True,
    generation_max_length=128,
    seed=42,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train
print(f"\n" + "=" * 80)
print(f"🚀 STARTING TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"=" * 80)
print(f"Expected time: 2-4 hours on A100")
print(f"Checkpoints will be saved to: {OUTPUT_DIR}")
print(f"=" * 80 + "\n")

try:
    train_results = trainer.train()
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {train_results.training_loss:.4f}")
    
    # Save model
    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅ Model saved to: {OUTPUT_DIR}")
    
    # Save summary
    summary = {
        'model': MODEL_NAME,
        'dataset': DATASET_NAME,
        'final_loss': float(train_results.training_loss),
        'output_dir': OUTPUT_DIR,
        'timestamp': datetime.now().isoformat(),
        'config': config,
    }
    
    with open(f'{OUTPUT_DIR}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print(f"✅ TRAINING PIPELINE COMPLETE")
print(f"=" * 80)
