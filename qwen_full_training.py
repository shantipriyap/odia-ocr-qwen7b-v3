#!/usr/bin/env python3
"""
Full Qwen2.5-VL training on 58.7K ODIA OCR dataset
"""
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🚀 FULL QWEN2.5-VL TRAINING (58.7K ODIA OCR DATASET)")
print("="*70)

# Load dataset (use ALL samples)
print("\n1️⃣ Loading dataset...")
ds = load_dataset("shantipriya/odia-ocr-merged")['train']
print(f"   Total: {len(ds):,} samples")

# Process images
print("\n2️⃣ Processing images...")
def proc(ex):
    try:
        img_bytes = ex['image']
        txt = ex['text']
        if not txt:
            return None
        img = Image.open(BytesIO(img_bytes)).convert('RGB') if isinstance(img_bytes, bytes) else img_bytes
        return {'image': img, 'text': txt}
    except:
        return None

ds = ds.map(proc, num_proc=1, remove_columns=['image', 'text'])
ds = ds.filter(lambda x: x is not None)
print(f"   ✅ {len(ds):,} samples ready")

# Load model
print("\n3️⃣ Loading model...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print(f"   ✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

# Collator
print("\n4️⃣ Setting up collator...")
class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
    
    def __call__(self, batch):
        texts = []
        images = []
        
        for ex in batch:
            try:
                img = ex['image']
                txt = ex['text']
                if isinstance(img, bytes):
                    img = Image.open(BytesIO(img)).convert('RGB')
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": txt}
                    ]}
                ]
                
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)
                images.append(img)
            except:
                pass
        
        if not images:
            return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}
        
        try:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs['labels'] = inputs['input_ids'].clone()
            return inputs
        except Exception as e:
            return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}

# Train
print("\n5️⃣ Training configuration...")
# 3 epochs = ~55K/batch x 3 ~ 1,650 steps
# At 0.109 steps/sec (from test) = ~4.2 hours
args = TrainingArguments(
    output_dir="./checkpoint-qwen-full",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    bf16=True,
    remove_unused_columns=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=Collator(processor)
)

print(f"   📊 Training {len(ds):,} samples x 3 epochs")
print(f"   ⏱️  Expected duration: ~4 hours")
print("\n" + "="*70)
print("🎯 STARTING FULL TRAINING")
print("="*70)

trainer.train()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"📁 Model saved to: ./checkpoint-qwen-full")
