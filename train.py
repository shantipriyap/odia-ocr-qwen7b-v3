#!/usr/bin/env python3
"""
Odia OCR Fine-Tuning Script - Qwen2.5-VL Model
=====================================================

Training script for fine-tuning Qwen2.5-VL-3B-Instruct on Odia OCR dataset.

Trains on 58,720 Odia text-image pairs for optical character recognition.
Uses gradient checkpointing and optimized memory management for A100 GPU.

Dataset: shantipriya/odia-ocr-merged
Model: Qwen/Qwen2.5-VL-3B-Instruct (3.78B parameters)
Hardware: NVIDIA A100-SXM4-80GB
"""

import torch
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from PIL import Image
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./checkpoint-odia-qwen"

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
MAX_STEPS = 3500  # 3 epochs on 58.7K samples
WARMUP_STEPS = 100
LEARNING_RATE = 2e-4
SAVE_STEPS = 500
LOGGING_STEPS = 50

logger.info("="*70)
logger.info("🚀 ODIA OCR FINE-TUNING: QWEN2.5-VL")
logger.info("="*70)
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Dataset: {DATASET_NAME}")
logger.info(f"Output: {OUTPUT_DIR}")
logger.info(f"Steps: {MAX_STEPS}")
logger.info("="*70)

# Load dataset
logger.info("\n📥 Loading dataset...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"✅ Loaded {len(dataset):,} samples")
except Exception as e:
    logger.error(f"❌ Failed to load dataset: {e}")
    raise

# Load model and processor
logger.info("\n📦 Loading model and processor...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("✅ Model and processor loaded")
    logger.info(f"   Device map: {model.device}")
    logger.info(f"   Dtype: {model.dtype}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# Enable gradient checkpointing for memory efficiency
logger.info("\n⚙️  Configuring gradient checkpointing...")
model.gradient_checkpointing_enable()
logger.info("✅ Gradient checkpointing enabled")

# Preprocess function
def preprocess_function(example):
    """Validate and prepare image-text pairs."""
    try:
        image = example.get("image")
        text = example.get("text", "")

        # Validate inputs
        if image is None or not text:
            return None

        # Ensure image is PIL Image
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                return None
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            return None

        return {"image": image, "text": text}
    except Exception:
        return None


# Custom data collator for OCR task
class OdiaOCRDataCollator:
    """Collate images and text for OCR training."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        """Process batch of examples."""
        images, texts = [], []

        for example in batch:
            try:
                if example is None:
                    continue

                img = example.get("image")
                text = example.get("text", "")

                # Validate
                if img is None or not text:
                    continue

                # Handle string paths
                if isinstance(img, str):
                    try:
                        img = Image.open(img).convert("RGB")
                    except Exception:
                        continue
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    continue

                images.append(img)
                texts.append(text)
            except Exception:
                continue

        # Return default if no valid examples
        if not images:
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]]),
            }

        # Process images and text
        try:
            inputs = self.processor(
                images,
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        except Exception as e:
            logger.warning(f"Error processing batch: {e}")
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]]),
            }


# Preprocess dataset
logger.info("\n🔄 Preprocessing dataset...")
try:
    train_dataset = dataset.map(
        preprocess_function, batched=False, num_proc=8, remove_columns=dataset.column_names
    )
    train_dataset = train_dataset.filter(lambda x: x is not None)
    logger.info(f"✅ {len(train_dataset):,} samples ready for training")
except Exception as e:
    logger.error(f"❌ Failed to preprocess dataset: {e}")
    raise

# Training arguments
logger.info("\n📋 Configuring training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=5,
    optim="adamw_torch",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    bf16=True,
    gradient_checkpointing=True,
    report_to=[],
    seed=42,
)

logger.info(f"✅ Batch size: {BATCH_SIZE} (accumulation x{GRADIENT_ACCUMULATION_STEPS})")
logger.info(f"   Learning rate: {LEARNING_RATE}")
logger.info(f"   Max steps: {MAX_STEPS}")
logger.info(f"   Save steps: {SAVE_STEPS}")

# Create trainer
logger.info("\n🎯 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=OdiaOCRDataCollator(processor),
)
logger.info("✅ Trainer initialized")

# Train
logger.info("\n" + "="*70)
logger.info("🚀 STARTING TRAINING")
logger.info("="*70)

try:
    trainer.train()
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"📁 Checkpoints saved to: {OUTPUT_DIR}")
except Exception as e:
    logger.error(f"\n❌ Training failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
    raise

logger.info("\n✨ Training pipeline complete!")
