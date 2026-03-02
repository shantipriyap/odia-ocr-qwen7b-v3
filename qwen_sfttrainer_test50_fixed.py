#!/usr/bin/env python3
"""
Fixed: Qwen2.5-VL with simple Trainer on 50 samples (no SFTTrainer complexity)
Addresses: Model class mismatch, message format issues, simpler data handling
"""

import torch
import logging
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModel, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.ERROR)

print("=" * 80)
print("🧪 FIXED TEST: QWEN2.5-VL (50 SAMPLES)")
print("=" * 80)

# ==================== CONFIG ====================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "shantipriya/odia-ocr-merged"
MAX_SAMPLES = 50
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_PIXELS = 1024 * 1024

print("\n📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("✅ Model loaded")

# ==================== LORA CONFIG ====================
print("\n⚙️  Configuring LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'v_proj']  # Simplified - only key modules
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== LOAD DATASET ====================
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_ID, split='train')
print(f"✅ Total: {len(dataset):,} samples")

# ==================== IMAGE RESIZING ====================
def resize_image(pil: Image.Image, max_side: int = MAX_IMAGE_SIDE, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    """Resize image to safe bounds"""
    pil = pil.convert("RGB")
    w, h = pil.size
    
    scale_side = min(1.0, max_side / float(max(w, h)))
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0
    scale = min(scale_side, scale_area)
    
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)
    
    return pil

# ==================== DATASET CLEANING (SINGLE PROCESS) ====================
print(f"\n🔄 Filtering first {MAX_SAMPLES} samples...")

def is_valid_sample(example):
    """Keep only valid image+text pairs"""
    image = example.get('image')
    text = (example.get('text') or '').strip()
    
    if not text or len(text) < 5:
        return False
    if image is None:
        return False
    return True

# Select first MAX_SAMPLES then filter
dataset_sample = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
dataset_clean = dataset_sample.filter(is_valid_sample, num_proc=1)
print(f"✅ Valid samples: {len(dataset_clean):,}/{MAX_SAMPLES}")

# ==================== PREPROCESS DATASET ====================
print("\n📋 Preprocessing data...")

def preprocess_fn(example):
    """Prepare image+text for training"""
    image = example['image']
    text = (example.get('text') or '').strip()
    
    # Load image if path
    if isinstance(image, str):
        try:
            image = Image.open(image).convert('RGB')
        except:
            return None
    elif hasattr(image, 'convert'):
        image = image.convert('RGB')
    else:
        return None
    
    # Resize
    image = resize_image(image)
    
    # Create simple prompt-response format
    prompt = "Extract all text from this Odia document image. Return only the extracted text."
    
    return {
        'image': image,
        'prompt': prompt,
        'text': text
    }

dataset_proc = dataset_clean.map(preprocess_fn, num_proc=1, remove_columns=[c for c in dataset_clean.column_names if c not in ['image', 'prompt', 'text']])
dataset_proc = dataset_proc.filter(lambda x: x is not None)
print(f"✅ Processed: {len(dataset_proc):,} samples")

# ==================== DATA COLLATOR ====================
class SimpleVLCollator:
    """Simple collator for vision-language training"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = [ex['image'] for ex in batch]
        texts = [f"{ex['prompt']}\n{ex['text']}" for ex in batch]
        
        try:
            enc = self.processor(
                text=texts,
                images=images,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Simple label assignment (no masking for now)
            enc['labels'] = enc['input_ids'].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            enc['labels'][enc['labels'] == pad_id] = -100
            
            return enc
        except Exception as e:
            print(f"⚠️  Collator error (batch skipped): {e}")
            return {
                'input_ids': torch.tensor([[0]]),
                'labels': torch.tensor([[0]])
            }

# ==================== TRAINING CONFIG ====================
print("\n📋 Training config (15 steps)...")

training_args = TrainingArguments(
    output_dir='./checkpoint-qwen-fixed-50',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_steps=15,
    warmup_steps=2,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=3,
    save_steps=15,
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
)

# ==================== CREATE TRAINER ====================
print("\n🎯 Creating trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_proc,
    data_collator=SimpleVLCollator(processor),
)

# ==================== TRAIN ====================
print("\n" + "=" * 80)
print("🚀 STARTING TEST TRAINING (15 STEPS)")
print("=" * 80 + "\n")

try:
    trainer.train()
    print("\n" + "=" * 80)
    print("✅ TEST TRAINING COMPLETE!")
    print("=" * 80)
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
