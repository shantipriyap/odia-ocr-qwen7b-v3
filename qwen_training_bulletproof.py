#!/usr/bin/env python3
"""
Bulletproof Qwen2.5-VL Training - NO UNSLOTH, NO SFTTrainer
Uses standard transformers Trainer with simple collator
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,  # ✅ CORRECT MODEL CLASS FOR VISION-LANGUAGE
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🚀 BULLETPROOF QWEN2.5-VL TRAINING (NO UNSLOTH)")
print("="*70)

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "shantipriya/odia-ocr-merged"
MAX_STEPS = 10  # Test on 10 samples first
BATCH_SIZE = 2
GRADIENT_ACC = 1

print(f"\n📋 CONFIG:")
print(f"   Model: {MODEL_ID}")
print(f"   Dataset: {DATASET_ID}")
print(f"   Steps: {MAX_STEPS}")
print(f"   Batch size: {BATCH_SIZE}")

# ============================================================================
# LOAD DATASET
# ============================================================================
print(f"\n📥 Loading dataset...")
dataset = load_dataset(DATASET_ID)
print(f"   ✅ {len(dataset['train']):,} samples available")

# ⚡ SELECT SAMPLES FIRST to avoid preprocessing entire dataset
if len(dataset['train']) > MAX_STEPS:
    dataset['train'] = dataset['train'].select(range(MAX_STEPS))
    print(f"   📊 Selected first {MAX_STEPS} samples for quick test")

# Convert to simple format: image bytes -> PIL Image, keep text
def load_images(example):
    """Convert image bytes to PIL Image"""
    try:
        image_bytes = example.get('image')
        text = example.get('text', '')
        
        if image_bytes is None or not text:
            return None
            
        # Convert bytes to PIL Image
        if isinstance(image_bytes, bytes):
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        else:
            image = image_bytes
            
        return {
            'image': image,
            'text': text
        }
    except Exception as e:
        print(f"   ⚠️  Error loading image: {e}")
        return None

print(f"\n🔄 Preprocessing (single-process)...")
train_dataset = dataset['train'].map(
    load_images, 
    num_proc=1,  # ✅ SINGLE PROCESS - no multiprocessing deadlock
    remove_columns=['image', 'text', 'category'] if 'category' in dataset['train'].column_names else ['image', 'text']
)
train_dataset = train_dataset.filter(lambda x: x is not None)
print(f"   ✅ {len(train_dataset):,} samples preprocessed")

# Take only first MAX_STEPS for testing
if len(train_dataset) > MAX_STEPS:
    train_dataset = train_dataset.select(range(MAX_STEPS))
    print(f"   📊 Using first {MAX_STEPS} samples for testing")

# ============================================================================
# LOAD MODEL & PROCESSOR
# ============================================================================
print(f"\n📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
print(f"   ✅ Processor loaded")

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # ✅ A100 native support
    device_map="auto",
    trust_remote_code=True
)
print(f"   ✅ Model loaded ({model.config.architectures[0]})")
print(f"   📊 Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ============================================================================
# APPLY LORA
# ============================================================================
print(f"\n⚙️  Applying LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================================
# DATA COLLATOR
# ============================================================================
class QwenVisionDataCollator:
    """Simple data collator for vision-language tasks"""
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        """Process batch of samples"""
        images = []
        texts = []
        
        for sample in batch:
            try:
                img = sample['image']
                text = sample['text']
                
                # Ensure PIL Image
                if isinstance(img, bytes):
                    img = Image.open(BytesIO(img)).convert('RGB')
                if not isinstance(img, Image.Image):
                    continue
                    
                images.append(img)
                texts.append(text)
            except Exception as e:
                print(f"   ⚠️  Collator error: {e}")
                continue
        
        if not images:
            # Return dummy batch
            return {
                'input_ids': torch.tensor([[0]]),
                'attention_mask': torch.tensor([[1]]),
                'pixel_values': torch.zeros(1, 3, 448, 448)
            }
        
        try:
            # Process with Qwen processor
            inputs = self.processor(
                images=images,
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Labels = input_ids for language modeling
            inputs['labels'] = inputs['input_ids'].clone()
            
            return inputs
        except Exception as e:
            print(f"   ❌ Processing error: {e}")
            return {
                'input_ids': torch.tensor([[0]]),
                'attention_mask': torch.tensor([[1]]),
                'pixel_values': torch.zeros(1, 3, 448, 448),
                'labels': torch.tensor([[0]])
            }

# ============================================================================
# TRAINING
# ============================================================================
print(f"\n📋 Setting up training...")
training_args = TrainingArguments(
    output_dir="./checkpoint-qwen-bulletproof",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACC,
    max_steps=MAX_STEPS,
    warmup_steps=5,
    logging_steps=1,
    save_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    bf16=True,  # ✅ Use bfloat16 on A100
    remove_unused_columns=False,
    report_to=[],
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=QwenVisionDataCollator(processor),
)

print(f"   ✅ Trainer ready")
print(f"   📊 Total steps: {len(train_dataset) // (BATCH_SIZE * GRADIENT_ACC)}")

# ============================================================================
# TRAIN
# ============================================================================
print(f"\n" + "="*70)
print(f"🚀 STARTING TRAINING")
print(f"="*70)

try:
    trainer.train()
    print(f"\n✅ TRAINING COMPLETE")
except KeyboardInterrupt:
    print(f"\n⏸️  Training interrupted")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"="*70)
