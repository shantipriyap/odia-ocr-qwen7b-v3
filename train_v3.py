#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import AutoProcessor, TrainingArguments, Trainer
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"

print("📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"✅ {len(dataset):,} samples")

print("📦 Loading model & processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

print("⚙️ Setup LoRA...")
lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def preprocess_fn(example):
    try:
        image = example.get("image")
        text = example.get("text", "").strip()
        if image is None or not text: return None
        if isinstance(image, str):
            try: image = Image.open(image).convert("RGB")
            except: return None
        elif hasattr(image, "convert"): image = image.convert("RGB")
        else: return None
        return {"image": image, "text": text}
    except: return None

print("🔄 Preprocessing...")
train_dataset = dataset.map(preprocess_fn, batched=False, num_proc=4, remove_columns=dataset.column_names)
train_dataset = train_dataset.filter(lambda x: x is not None)
print(f"✅ {len(train_dataset):,} ready")

class OCRCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images, texts = [], []
        for item in batch:
            img = item.get("image")
            txt = item.get("text", "").strip()
            if img and txt:
                images.append(img)
                texts.append(txt)
        
        if not images:
            return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]]), "labels": torch.tensor([[-100]])}
        
        # Core fix: text FIRST, images SECOND
        inputs = self.processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["input_ids"] == 0] = -100
        return inputs

print("📋 Training config...")
training_args = TrainingArguments(
    output_dir="./checkpoint-a100-v3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=500,
    warmup_steps=50,
    logging_steps=5,
    save_steps=50,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    optim="adamw_torch",
    bf16=True,
    report_to=[],
    seed=42,
)

print("🎯 Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=OCRCollator(processor),
)

print("=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)

trainer.train()

print("✅ COMPLETE")
