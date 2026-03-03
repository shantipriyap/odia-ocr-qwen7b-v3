#!/usr/bin/env python3
"""
Simplest version - only process new odia data
2 columns: text and image
"""

import os, sys, gc
from datasets import Dataset, DatasetDict, Features, Value, Image as DatasetImage, concatenate_datasets
from PIL import Image
from tqdm import tqdm

# Config
DATA_PATH = "/Users/shantipriya/work/odia_ocr/odia"
LABELS_FILE = f"{DATA_PATH}/train.txt"
REPO = "shantipriya/odia-ocr-merged"
BATCH = 500
SEED = 42

print("="*60, flush=True)
print("NEW ODIA DATASET - text + image only", flush=True)
print("="*60, flush=True)

features = Features({'text': Value('string'), 'image': DatasetImage()})

# Read labels
print(f"\nReading labels...", flush=True)
with open(LABELS_FILE) as f:
    lines = f.readlines()
print(f"Total lines: {len(lines):,}", flush=True)

# Process in batches
print(f"\nProcessing in batches of {BATCH}...", flush=True)
batches = []
processed = skipped = 0

for i in range(0, len(lines), BATCH):
    batch_lines = lines[i:i+BATCH]
    data = {'text': [], 'image': []}
    
    for line in batch_lines:
        parts = line.strip().split(' ', 1)
        if len(parts) != 2:
            skipped += 1
            continue
        
        img_path, text = parts
        if not text.strip():
            skipped += 1
            continue
        
        full_path = f"{DATA_PATH}/{img_path}"
        if not os.path.exists(full_path):
            skipped += 1
            continue
        
        try:
            with Image.open(full_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                data['text'].append(text.strip())
                data['image'].append(img.copy())
                processed += 1
        except:
            skipped += 1
    
    if len(data['text']) > 0:
        batches.append(Dataset.from_dict(data, features=features))
        print(f"  Batch {len(batches)}: {len(data['text'])} samples", flush=True)
    
    del data
    gc.collect()

print(f"\n✅ Processed: {processed:,}", flush=True)
print(f"⚠️  Skipped: {skipped:,}", flush=True)

# Concatenate
print(f"\nConcatenating {len(batches)} batches...", flush=True)
ds = concatenate_datasets(batches)
del batches
gc.collect()
print(f"Total: {len(ds):,} samples", flush=True)

# Split
print(f"\nCreating train/val/test splits...", flush=True)
ds = ds.shuffle(seed=SEED)
total = len(ds)
train_size = int(total * 0.8)
val_size = int(total * 0.1)

splits = DatasetDict({
    'train': ds.select(range(train_size)),
    'validation': ds.select(range(train_size, train_size + val_size)),
    'test': ds.select(range(train_size + val_size, total))
})

print(f"  Train: {len(splits['train']):,}", flush=True)
print(f"  Val: {len(splits['validation']):,}", flush=True)
print(f"  Test: {len(splits['test']):,}", flush=True)

del ds
gc.collect()

# Upload
print(f"\nUploading to {REPO}...", flush=True)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not set")
    sys.exit(1)

try:
    splits.push_to_hub(REPO, token=hf_token, 
                      commit_message=f"New Odia OCR: {total:,} samples, text+image only, train/val/test splits")
    print(f"\n{'='*60}", flush=True)
    print(f"✅ SUCCESS! {total:,} samples uploaded", flush=True)
    print(f"🔗 https://huggingface.co/datasets/{REPO}", flush=True)
    print(f"{'='*60}", flush=True)
except Exception as e:
    print(f"❌ Upload failed: {e}", flush=True)
    sys.exit(1)
