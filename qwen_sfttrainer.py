#!/usr/bin/env python3
"""
Qwen2.5-VL training using TRL's SFTTrainer (bulletproof approach from DataCamp)
"""

import torch
import logging
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.ERROR)

print("=" * 80)
print("🚀 QWEN2.5-VL TRAINING (DATA CAMP METHOD)")
print("=" * 80)

# ==================== CONFIG ====================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "shantipriya/odia-ocr-merged"
MAX_TARGET_CHARS = 500
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_PIXELS = 1024 * 1024

print("\n📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
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
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
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
print("\n🔄 Filtering dataset (single process - no hangs)...")

def is_valid_sample(example):
    """Keep only valid image+text pairs"""
    image = example.get('image')
    text = (example.get('text') or '').strip()
    
    if not text or len(text) < 5:
        return False
    if image is None:
        return False
    return True

# Filter WITHOUT multiprocessing
dataset_clean = dataset.filter(is_valid_sample, num_proc=1)
print(f"✅ Valid samples: {len(dataset_clean):,}")

# ==================== CONVERT TO MESSAGES ====================
print("\n📋 Converting to message format...")

def to_messages(example):
    """Convert to chat message format"""
    image = example['image']
    text = (example.get('text') or '').strip()
    
    # Ensure image is PIL
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
    
    # Build prompt and target
    prompt = "Extract all text from this Odia document image. Return only the extracted text."
    target = text[:MAX_TARGET_CHARS]
    
    example['messages'] = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]
        },
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': target}]
        }
    ]
    example['image'] = image
    return example

# Map WITHOUT multiprocessing
dataset_messages = dataset_clean.map(to_messages, num_proc=1, remove_columns=[c for c in dataset_clean.column_names if c not in ['image', 'messages']])
dataset_messages = dataset_messages.filter(lambda x: x is not None)
print(f"✅ Processed: {len(dataset_messages):,} samples")

# ==================== DATA COLLATOR ====================
class OdiaCollator:
    """Custom collator for vision-language training"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        # Extract texts at full chat length
        full_texts = [
            self.processor.apply_chat_template(
                ex['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
            for ex in batch
        ]
        
        # Extract prompt-only texts (for loss masking)
        prompt_texts = [
            self.processor.apply_chat_template(
                ex['messages'][:-1],
                tokenize=False,
                add_generation_prompt=True
            )
            for ex in batch
        ]
        
        images = [ex['image'] for ex in batch]
        
        # Tokenize with images
        try:
            enc = self.processor(
                text=full_texts,
                images=images,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            )
        except Exception as e:
            print(f"Collator error: {e}")
            return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}
        
        input_ids = enc['input_ids']
        pad_id = self.processor.tokenizer.pad_token_id
        
        # Get prompt lengths (text-only for speed)
        prompt_ids = self.processor.tokenizer(
            prompt_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False
        )['input_ids']
        
        prompt_lens = (prompt_ids != pad_id).sum(dim=1)
        
        # Create labels: mask prompt + padding
        labels = input_ids.clone()
        bs, seqlen = labels.shape
        
        for i in range(bs):
            pl = int(prompt_lens[i].item())
            pl = min(pl, seqlen)
            labels[i, :pl] = -100
        
        labels[labels == pad_id] = -100
        enc['labels'] = labels
        
        return enc

# ==================== TRAINING CONFIG ====================
print("\n📋 Setting up training...")

training_args = SFTConfig(
    output_dir='./checkpoint-qwen-odia-sft',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=50,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    lr_scheduler_type='cosine',
    logging_steps=10,
    report_to=[],
    remove_unused_columns=False,
    gradient_checkpointing=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
)

# ==================== CREATE TRAINER ====================
print("\n🎯 Creating SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_messages,
    data_collator=OdiaCollator(processor),
    peft_config=lora_config
)

# ==================== TRAIN ====================
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80 + "\n")

try:
    trainer.train()
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
