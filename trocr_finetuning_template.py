#!/usr/bin/env python3
"""
TrOCR Fine-tuning on Odia Dataset
Fine-tune TrOCR on your 145K Odia OCR dataset for better accuracy
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from PIL import Image as PILImage
import os

print("=" * 80)
print("🚀 TROCR FINE-TUNING ON ODIA OCR DATASET")
print("=" * 80)

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"

print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"✅ Dataset: {len(dataset):,} samples")

print("\n📦 Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

def preprocess_images(examples):
    """Preprocess images and text"""
    images = []
    labels = []
    
    for idx, item in enumerate(examples):
        try:
            img = item.get('image')
            txt = item.get('text', '').strip()
            
            if not img or not txt:
                continue
                
            if isinstance(img, str):
                img = PILImage.open(img).convert('RGB')
            else:
                img = img.convert('RGB')
            
            images.append(img)
            labels.append(txt)
        except:
            continue
    
    if not images:
        return {"pixel_values": [], "labels": []}
    
    # Process images
    pixel_values = processor(images, return_tensors="pt").pixel_values
    
    # Process text
    encoding = processor.tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
    
    return {
        "pixel_values": pixel_values,
        "labels": encoding.input_ids
    }

print("\n🔄 Preprocessing dataset...")
# Take subset for demo (use full dataset for production)
train_dataset = dataset.select(range(min(1000, len(dataset))))
train_dataset = train_dataset.map(preprocess_images, batched=True, batch_size=32)

print(f"✅ Preprocessed: {len(train_dataset)} samples")

print("\n📋 Training config...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-odia-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    learning_rate=5e-5,
    warmup_steps=100,
    lr_scheduler_type="linear",
    save_total_limit=3,
    predict_with_generate=True,
    eval_strategy="no",
    gradient_accumulation_steps=2,
    fp16=True,
    dataloader_num_workers=4,
)

print("🎯 Creating trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,
)

print("\n" + "=" * 80)
print("🚀 STARTING FINE-TUNING")
print("=" * 80)

try:
    trainer.train()
    print("\n✅ FINE-TUNING COMPLETE")
    
    # Save
    print("\n💾 Saving model...")
    model.save_pretrained("./trocr-odia-finetuned")
    processor.save_pretrained("./trocr-odia-finetuned")
    print("✅ Model saved to ./trocr-odia-finetuned")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
