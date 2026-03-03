#!/usr/bin/env python3
"""
🧪 Test TrOCR Training with Few Samples
Diagnose data issues before full training
"""

import torch
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("🧪 TESTING TROCR WITH 10 SAMPLES")
print("=" * 80)

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "shantipriya/odia-ocr-merged"

print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_data = dataset['train']
print(f"✅ Dataset loaded: {len(train_data)} total samples")

# Take only 10 samples for testing
test_dataset = train_data.select(range(min(10, len(train_data))))
print(f"✅ Selected {len(test_dataset)} samples for testing")

print("\n📦 Loading model and tokenizer...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
print(f"✅ Model loaded: {MODEL_NAME}")
print(f"   Device: {next(model.parameters()).device}")

print("\n⚙️  Adding LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none'
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n🔄 Testing preprocessing...")

def preprocess_sample(example):
    try:
        image = example['image']
        text = example['text'].strip()
        
        if not text or len(text) < 1:
            return None
        
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        return {'image': image, 'text': text}
    except Exception as e:
        print(f"   ⚠️  Preprocessing error: {e}")
        return None

# Process
processed = []
for i, sample in enumerate(test_dataset):
    result = preprocess_sample(sample)
    if result:
        processed.append(result)
        print(f"  ✓ Sample {i+1}: Image {result['image'].size}, Text: {result['text'][:30]}...")

print(f"✅ Successfully processed {len(processed)} samples")

if len(processed) < 2:
    print("❌ Not enough valid samples to test")
    exit(1)

# Test data collator
print("\n📋 Testing data collator...")

class TrOCRDataCollator:
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        images = [ex['image'] for ex in batch]
        texts = [ex['text'] for ex in batch]
        
        # Process images
        pixel_values = self.processor(images=images, return_tensors='pt')['pixel_values']
        print(f"  ✓ Images processed: {pixel_values.shape}")
        
        # Tokenize text
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        print(f"  ✓ Text tokenized: input_ids shape {encodings['input_ids'].shape}")
        
        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': encodings['input_ids'],
            'labels': encodings['input_ids'].clone(),
        }

collator = TrOCRDataCollator(processor, tokenizer)
batch = collator(processed[:2])
print(f"✅ Batch created successfully")
print(f"   pixel_values: {batch['pixel_values'].shape}")
print(f"   decoder_input_ids: {batch['decoder_input_ids'].shape}")
print(f"   labels: {batch['labels'].shape}")

# Test forward pass
print("\n🔮 Testing forward pass...")
try:
    model = model.cuda()
    batch_gpu = {k: v.cuda() for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch_gpu)
    
    print(f"✅ Forward pass successful!")
    print(f"   Loss shape: {outputs.loss.shape if outputs.loss is not None else 'None'}")
    print(f"   Loss value: {outputs.loss.item() if outputs.loss is not None else 'None'}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - READY FOR FULL TRAINING")
print("=" * 80)
