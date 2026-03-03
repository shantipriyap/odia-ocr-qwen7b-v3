#!/usr/bin/env python3
"""
Simple merge with only text and image columns
Validates both fields are present before adding
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
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

print("="*80)
print("📊 SIMPLIFIED DATASET MERGE (text + image only)")
print("="*80)

# Define simple schema - only text and image
features = Features({
    'text': Value('string'),
    'image': DatasetImage()
})

# Step 1: Load and clean existing dataset
print("\n[1/5] 📥 Loading existing dataset...")
try:
    existing_full = load_dataset(EXISTING_DATASET, split="train")
    print(f"   Loaded {len(existing_full):,} existing samples")
    
    # Extract only text and image, skip if either is missing
    print(f"   Filtering for valid text and image...")
    valid_existing = []
    
    for i in tqdm(range(len(existing_full)), desc="   Validating"):
        item = existing_full[i]
        text = item.get('text')
        image = item.get('image')
        
        # Skip if text is None/empty or image is None
        if text and text.strip() and image is not None:
            valid_existing.append({
                'text': text.strip(),
                'image': image
            })
    
    print(f"   ✅ Valid existing samples: {len(valid_existing):,}")
    print(f"   ⚠️  Filtered out: {len(existing_full) - len(valid_existing):,}")
    
    existing_ds = Dataset.from_dict({
        'text': [x['text'] for x in valid_existing],
        'image': [x['image'] for x in valid_existing]
    }, features=features)
    
    del existing_full, valid_existing
    gc.collect()
    
except Exception as e:
    print(f"   ⚠️  Could not load existing dataset: {e}")
    existing_ds = None

# Step 2: Process new data
print(f"\n[2/5] 🔄 Processing new dataset...")

with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Total lines in file: {len(lines):,}")

valid_new = []
skipped = 0

for line in tqdm(lines, desc="   Processing"):
    parts = line.strip().split(' ', 1)
    
    # Validate format
    if len(parts) != 2:
        skipped += 1
        continue
    
    img_path, text = parts
    
    # Validate text is not empty
    if not text or not text.strip():
        skipped += 1
        continue
    
    full_path = f"{NEW_DATA_PATH}/{img_path}"
    
    # Validate image exists
    if not os.path.exists(full_path):
        skipped += 1
        continue
    
    try:
        # Load and validate image
        with Image.open(full_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_copy = img.copy()
        
        # Both text and image are valid
        valid_new.append({
            'text': text.strip(),
            'image': img_copy
        })
        
    except Exception as e:
        skipped += 1
        continue

print(f"   ✅ Valid new samples: {len(valid_new):,}")
print(f"   ⚠️  Skipped: {skipped:,}")

# Create new dataset
print(f"   Creating dataset from valid samples...")
new_ds = Dataset.from_dict({
    'text': [x['text'] for x in valid_new],
    'image': [x['image'] for x in valid_new]
}, features=features)

del valid_new
gc.collect()

print(f"   ✅ New dataset: {len(new_ds):,} samples")

# Step 3: Merge datasets
print(f"\n[3/5] 🔗 Merging datasets...")

if existing_ds:
    print(f"   Merging {len(existing_ds):,} + {len(new_ds):,} samples...")
    merged_ds = concatenate_datasets([existing_ds, new_ds])
    del existing_ds, new_ds
    gc.collect()
else:
    merged_ds = new_ds

total = len(merged_ds)
print(f"   ✅ Total: {total:,} samples (all validated)")

# Step 4: Create splits
print(f"\n[4/5] ✂️  Creating train/dev/test splits...")

# Shuffle
merged_ds = merged_ds.shuffle(seed=RANDOM_SEED)

# Calculate splits
train_size = int(total * TRAIN_RATIO)
dev_size = int(total * DEV_RATIO)
test_size = total - train_size - dev_size

print(f"   Split sizes:")
print(f"      - Train:      {train_size:,} ({train_size/total*100:.1f}%)")
print(f"      - Validation: {dev_size:,} ({dev_size/total*100:.1f}%)")
print(f"      - Test:       {test_size:,} ({test_size/total*100:.1f}%)")

# Create splits
train_ds = merged_ds.select(range(train_size))
val_ds = merged_ds.select(range(train_size, train_size + dev_size))
test_ds = merged_ds.select(range(train_size + dev_size, total))

dataset_dict = DatasetDict({
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
})

del merged_ds, train_ds, val_ds, test_ds
gc.collect()

print(f"   ✅ Splits created")

# Step 5: Upload
print(f"\n[5/5] 📤 Uploading to HuggingFace...")
print(f"   Repository: {OUTPUT_DATASET_REPO}")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not set")
    exit(1)

try:
    print(f"   ⏳ Uploading {total:,} samples with only text + image columns...")
    dataset_dict.push_to_hub(
        OUTPUT_DATASET_REPO,
        token=hf_token,
        commit_message=f"Simplified dataset: text + image only, {total:,} validated samples with train/dev/test splits"
    )
    
    print(f"\n✅ Upload successful!")
    print(f"   🔗 https://huggingface.co/datasets/{OUTPUT_DATASET_REPO}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"   ✅ Dataset with 2 columns: text, image")
    print(f"   ✅ All entries validated (non-empty text + valid image)")
    print(f"\n   Total samples: {total:,}")
    print(f"\n   Splits:")
    print(f"      - train:      {len(dataset_dict['train']):,}")
    print(f"      - validation: {len(dataset_dict['validation']):,}")
    print(f"      - test:       {len(dataset_dict['test']):,}")
    print(f"\n   Usage:")
    print(f"      from datasets import load_dataset")
    print(f"      dataset = load_dataset('{OUTPUT_DATASET_REPO}')")
    print(f"      # Each sample has: {{'text': '...', 'image': <PIL.Image>}}")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
