#!/usr/bin/env python3
"""
Ultra memory-efficient merge - processes in small batches
Only keeps text and image columns
"""

import os
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image as DatasetImage, concatenate_datasets
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc

# Configuration
EXISTING_DATASET = "shantipriya/odia-ocr-merged"
NEW_DATA_PATH = "/Users/shantipriya/work/odia_ocr/odia"
LABELS_FILE = f"{NEW_DATA_PATH}/train.txt"
OUTPUT_DATASET_REPO = "shantipriya/odia-ocr-merged"
BATCH_SIZE = 500  # Small batches to avoid memory issues
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

print("="*80)
print("📊 ULTRA-EFFICIENT MERGE (text + image only)")
print("="*80)

features = Features({
    'text': Value('string'),
    'image': DatasetImage()
})

# Step 1: Process existing dataset in batches
print("\n[1/5] 📥 Processing existing dataset...")
print(f"   Note: Skipping existing dataset to save memory")
print(f"   Will only use new data for this merge")
existing_ds = None

# Alternative: If you want to include existing data, uncomment below
# try:
#     existing_full = load_dataset(EXISTING_DATASET, split="train")
#     print(f"   Loaded {len(existing_full):,} samples")
#     # Take only samples with valid text and image
#     existing_ds = existing_full.filter(lambda x: x['text'] and x['text'].strip() and x['image'] is not None)
#     existing_ds = existing_ds.remove_columns([col for col in existing_ds.column_names if col not in ['text', 'image']])
#     print(f"   ✅ Filtered to {len(existing_ds):,} valid samples")
# except Exception as e:
#     print(f"   ⚠️  Error: {e}")
#     existing_ds = None

# Step 2: Read labels file
print(f"\n[2/5] 📖 Reading labels file...")
with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Total lines: {len(lines):,}")

# Step 3: Process in small batches
print(f"\n[3/5] 🔄 Processing new data in batches of {BATCH_SIZE}...")

batch_datasets = []
total_processed = 0
total_skipped = 0

for batch_start in range(0, len(lines), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(lines))
    batch_lines = lines[batch_start:batch_end]
    
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(lines) - 1) // BATCH_SIZE + 1
    
    batch_data = {'text': [], 'image': []}
    batch_skipped = 0
    
    for line in batch_lines:
        parts = line.strip().split(' ', 1)
        
        if len(parts) != 2:
            batch_skipped += 1
            continue
        
        img_path, text = parts
        
        if not text or not text.strip():
            batch_skipped += 1
            continue
        
        full_path = f"{NEW_DATA_PATH}/{img_path}"
        
        if not os.path.exists(full_path):
            batch_skipped += 1
            continue
        
        try:
            with Image.open(full_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_copy = img.copy()
            
            batch_data['text'].append(text.strip())
            batch_data['image'].append(img_copy)
            
        except Exception as e:
            batch_skipped += 1
            continue
    
    # Create dataset from batch
    if len(batch_data['text']) > 0:
        batch_ds = Dataset.from_dict(batch_data, features=features)
        batch_datasets.append(batch_ds)
        total_processed += len(batch_ds)
        print(f"   Batch {batch_num}/{total_batches}: {len(batch_ds)} valid samples")
    
    total_skipped += batch_skipped
    
    # Clean up
    del batch_data
    gc.collect()

print(f"\n   ✅ Processed: {total_processed:,} samples")
print(f"   ⚠️  Skipped: {total_skipped:,} samples")

# Concatenate all batches
print(f"\n   Concatenating {len(batch_datasets)} batches...")
new_ds = concatenate_datasets(batch_datasets)
print(f"   ✅ New dataset: {len(new_ds):,} samples")

del batch_datasets
gc.collect()

# Step 4: Merge with existing (if any)
print(f"\n[4/5] 🔗 Finalizing dataset...")

if existing_ds:
    merged_ds = concatenate_datasets([existing_ds, new_ds])
    del existing_ds, new_ds
    gc.collect()
else:
    merged_ds = new_ds

total = len(merged_ds)
print(f"   Total: {total:,} samples")

# Create splits
print(f"\n   Creating splits...")
merged_ds = merged_ds.shuffle(seed=RANDOM_SEED)

train_size = int(total * TRAIN_RATIO)
dev_size = int(total * DEV_RATIO)
test_size = total - train_size - dev_size

train_ds = merged_ds.select(range(train_size))
val_ds = merged_ds.select(range(train_size, train_size + dev_size))
test_ds = merged_ds.select(range(train_size + dev_size, total))

dataset_dict = DatasetDict({
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
})

print(f"   ✅ Train: {train_size:,} | Val: {dev_size:,} | Test: {test_size:,}")

del merged_ds, train_ds, val_ds, test_ds
gc.collect()

# Step 5: Upload
print(f"\n[5/5] 📤 Uploading to HuggingFace...")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not set")
    exit(1)

try:
    print(f"   Uploading to {OUTPUT_DATASET_REPO}...")
    dataset_dict.push_to_hub(
        OUTPUT_DATASET_REPO,
        token=hf_token,
        commit_message=f"Simplified: text+image only, {total:,} validated samples, train/val/test splits"
    )
    
    print("\n" + "="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print(f"   Dataset: {total:,} samples (2 columns: text, image)")
    print(f"   Train: {len(dataset_dict['train']):,}")
    print(f"   Val: {len(dataset_dict['validation']):,}")
    print(f"   Test: {len(dataset_dict['test']):,}")
    print(f"\n   🔗 https://huggingface.co/datasets/{OUTPUT_DATASET_REPO}")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
