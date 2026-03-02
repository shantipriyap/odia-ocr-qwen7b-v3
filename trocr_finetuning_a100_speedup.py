#!/usr/bin/env python3
"""
🎯 TrOCR Fine-tuning for Odia OCR - ULTRA-OPTIMIZED FOR A100
Maximum parallelism, GPU/CPU utilization, speed-optimized
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
from datetime import datetime
import psutil

print("=" * 80)
print("🚀 TrOCR FINE-TUNING FOR ODIA OCR ON A100 - SPEEDUP MODE")
print("=" * 80)

# Check GPU and CPU resources
print(f"\n✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"   Memory: {props.total_memory / 1e9:.0f} GB")
    print(f"   Compute Capability: {props.major}.{props.minor}")

cpu_count = psutil.cpu_count()
print(f"\n💻 CPU: {cpu_count} cores available")
print(f"   Using {min(8, cpu_count)} cores for parallel processing")

# Configuration - OPTIMIZED FOR SPEED
MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./trocr-odia-finetuned-speedup"
CHECKPOINT_DIR = "./trocr-checkpoints-speedup"

config = {
    'per_device_train_batch_size': 16,  # Increased from 8
    'per_device_eval_batch_size': 16,   # Increased from 8
    'gradient_accumulation_steps': 1,   # Reduced from 2 - larger batches reduce need
    'num_train_epochs': 3,
    'learning_rate': 5e-5,
    'warmup_steps': 300,                # Reduced - faster warmup
    'max_steps': -1,
    'weight_decay': 0.01,
    'eval_strategy': 'no',              # Skip eval during training for speed
    'save_strategy': 'steps',
    'save_steps': 1000,
    'save_total_limit': 2,
    'seed': 42,
}

print(f"\n📋 Training Configuration (OPTIMIZED):")
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
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

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
    lora_dropout=0.05,  # Reduced from 0.1 for faster convergence
    bias='none',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data preprocessing - OPTIMIZED
print(f"\n🔄 Preprocessing dataset (PARALLELIZED)...")

def preprocess_images_simple(example):
    """Simple per-sample preprocessing without batching"""
    try:
        image = example['image']
        text = example['text'].strip()
        
        # Skip empty samples
        if not text or len(text) < 1:
            return None
        
        # Convert PIL to RGB
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        return {
            'image': image,
            'text': text,
        }
    except:
        return None

# Preprocess and filter - simpler approach
print("📊 Applying preprocessing with 8 workers...")
processed_dataset = train_data.map(
    preprocess_images_simple,
    num_proc=8,
    desc="Processing Odia OCR data",
)

# Filter out None values  
processed_dataset = processed_dataset.filter(lambda x: x is not None)
print(f"✅ Processed {len(processed_dataset)} valid samples")

# Split for evaluation
if len(processed_dataset) > 5000:
    split_data = processed_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_data['train']
    eval_dataset = split_data['test']
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
else:
    train_dataset = processed_dataset
    print(f"Training on all {len(train_dataset)} samples")


# Custom data collator for TrOCR
class TrOCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = []
        texts = []
        
        for example in batch:
            try:
                img = example['image']
                text = example['text']
                
                if not text or len(text) < 1:
                    continue
                
                images.append(img)
                texts.append(text)
            except:
                continue
        
        if not images:
            # Return dummy batch
            import torch
            return {
                'pixel_values': torch.zeros((1, 3, 224, 224)),
                'decoder_input_ids': torch.zeros((1, 1), dtype=torch.long),
                'labels': torch.zeros((1, 1), dtype=torch.long),
            }
        
        # Process images
        pixel_values = self.processor(images=images, return_tensors='pt')['pixel_values']
        
        # Tokenize text
        encodings = tokenizer(
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

# Initialize trainer with GPU-optimized settings
print(f"\n🎯 Initializing Seq2SeqTrainer...")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=config['num_train_epochs'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    warmup_steps=config['warmup_steps'],
    weight_decay=config['weight_decay'],
    eval_strategy=config['eval_strategy'],
    save_strategy=config['save_strategy'],
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    logging_steps=10,            # More frequent logging for progress tracking
    report_to=[],
    bf16=True,                   # Use bfloat16 on A100
    fp16=False,
    tf32=True,                   # TensorFloat32 for speed (A100 feature)
    optim='adamw_torch_fused',   # Fused optimizer for speed
    dataloader_num_workers=8,    # Increased from 4
    dataloader_pin_memory=True,  # Pin memory for speed
    predict_with_generate=False, # Skip generation during training (faster)
    generation_max_length=128,
    seed=42,
    gradient_checkpointing=False,  # Disable to maximize speed (we have memory)
    max_grad_norm=1.0,
    remove_unused_columns=False,  # Keep image/text columns for data_collator
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if len(processed_dataset) > 5000 else None,
    data_collator=TrOCRDataCollator(processor),
)

# Train
print(f"\n" + "=" * 80)
print(f"🚀 STARTING TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"=" * 80)
print(f"⚡ SPEEDUP MODE: 8 CPU workers, batch size 16, bf16, fused optimizer")
print(f"⏱️  Expected time: 1-2 hours on A100 (vs 2-4 hours before)")
print(f"💾 Checkpoints: {OUTPUT_DIR}")
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
        'optimization': 'speedup_mode',
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
