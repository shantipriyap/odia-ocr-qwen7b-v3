#!/usr/bin/env python3
"""
🎯 Continue Training Qwen2.5-Odia-OCR-v2 with New Data
Load existing model from HuggingFace and fine-tune with fresh odia-ocr-merged dataset
"""

import torch
import json
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datetime import datetime
import os

print("=" * 80)
print("🎯 QWEN2.5-ODIA-OCR CONTINUATION TRAINING")
print("=" * 80)

# Model and dataset config
MODEL_NAME = "shantipriya/qwen2.5-odia-ocr-v2"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./checkpoint-qwen-odia-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n📦 Configuration:")
print(f"   Base Model: {MODEL_NAME}")
print(f"   New Dataset: {DATASET_NAME}")
print(f"   Device: {DEVICE}")
print(f"   Output: {OUTPUT_DIR}")

# Load model and processor
print("\n📦 Loading pre-trained model...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print(f"✅ Model loaded: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Apply LoRA for efficient training
print("\n⚙️  Configuring LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print(f"\n📥 Loading {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME)
print(f"✅ Loaded {len(dataset['train']):,} samples")

# Split into train/eval
train_test_split = dataset['train'].train_test_split(test_size=0.02, seed=42)
train_data = train_test_split['train']
eval_data = train_test_split['test']
print(f"✅ Train: {len(train_data):,} | Eval: {len(eval_data):,}")

# Preprocessing function
def preprocess_function(example):
    try:
        image = example['image']
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        text = example.get('text', '').strip()
        if not text:
            return None
        
        return {'image': image, 'text': text}
    except:
        return None

print("🔄 Preprocessing training data...")
train_data = train_data.map(preprocess_function, batched=False, num_proc=8)
train_data = train_data.filter(lambda x: x is not None)
print(f"✅ {len(train_data):,} training samples ready")

print("🔄 Preprocessing eval data...")
eval_data = eval_data.map(preprocess_function, batched=False, num_proc=8)
eval_data = eval_data.filter(lambda x: x is not None)
print(f"✅ {len(eval_data):,} eval samples ready")

# Data collator for Qwen
class QwenOCRDataCollator:
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
                    from PIL import Image
                    img = Image.open(img).convert('RGB')
                elif hasattr(img, 'convert'):
                    img = img.convert('RGB')
                else:
                    continue
                
                if not text:
                    continue
                
                images.append(img)
                texts.append(text)
            except:
                continue
        
        if not images:
            # Return dummy batch to avoid errors
            return {
                'input_ids': torch.tensor([[1]]),
                'attention_mask': torch.tensor([[1]]),
                'labels': torch.tensor([[1]]),
            }
        
        # Process with Qwen processor
        try:
            # Format text with Qwen's OCR instruction
            formatted_texts = [f"Extract the Odia text from this image:\n{text}" for text in texts]
            
            inputs = self.processor(
                images=images,
                text=formatted_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            
            # Set labels for language modeling
            inputs['labels'] = inputs['input_ids'].clone()
            
            return inputs
        except Exception as e:
            print(f"⚠️  Collation error: {e}")
            return {
                'input_ids': torch.tensor([[1]]),
                'attention_mask': torch.tensor([[1]]),
                'labels': torch.tensor([[1]]),
            }

collator = QwenOCRDataCollator(processor)

# Training arguments
print("\n📋 Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=100,
    save_total_limit=2,
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=collator,
)
print("✅ Trainer ready")

# Start training
print("\n" + "=" * 80)
print("🚀 STARTING CONTINUATION TRAINING")
print("=" * 80)

try:
    train_result = trainer.train()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    print(f"   Total steps: {train_result.global_step}")
    
    # Save final model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Model saved!")
    
    # Upload to HuggingFace
    print("\n📤 Preparing for HuggingFace upload...")
    print(f"   Run: huggingface-cli upload shantipriya/qwen2.5-odia-ocr-v3 {OUTPUT_DIR}/*")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
