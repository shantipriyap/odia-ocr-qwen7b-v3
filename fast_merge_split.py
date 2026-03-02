#!/usr/bin/env python3
"""
Fast merge with train/dev/test splits - uses parquet for efficiency
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
print("📊 FAST ODIA DATASET MERGE & SPLIT")
print("="*80)

# Step 1: Load existing dataset
print("\n[1/5] 📥 Loading existing dataset from HuggingFace...")
try:
    existing_ds = load_dataset(EXISTING_DATASET, split="train")
    print(f"✅ Loaded {len(existing_ds):,} existing samples")
    existing_count = len(existing_ds)
except Exception as e:
    print(f"⚠️  Could not load existing dataset: {e}")
    existing_ds = None
    existing_count = 0

# Step 2: Process new data efficiently
print(f"\n[2/5] 🔄 Processing new dataset...")

with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Reading {len(lines):,} samples...")

# Prepare data (without loading images yet)
new_data = []
skipped = 0

for line in tqdm(lines, desc="   Parsing labels"):
    parts = line.strip().split(' ', 1)
    if len(parts) != 2:
        skipped += 1
        continue
    
    img_path, text = parts
    full_path = f"{NEW_DATA_PATH}/{img_path}"
    
    if not os.path.exists(full_path):
        skipped += 1
        continue
    
    new_data.append({
        'path': full_path,
        'img_path': img_path,
        'text': text,
        'filename': os.path.basename(img_path)
    })

print(f"   ✅ Prepared {len(new_data):,} samples (skipped {skipped:,})")

# Step 3: Create new dataset with generator (more memory efficient)
print(f"\n[3/5] 🔨 Creating new dataset...")

def generate_examples():
    """Generator to yield examples one at a time"""
    for item in tqdm(new_data, desc="   Loading images"):
        try:
            with Image.open(item['path']) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_copy = img.copy()
            
            yield {
                'text': item['text'],
                'image': img_copy,
                'image_path': item['img_path'],
                'filename': item['filename'],
                'class_id': None,
                'character': None,
                'type': "word",
                'augmentation': None
            }
        except Exception as e:
            print(f"\n      ⚠️  Error loading {item['path']}: {e}")
            continue

features = Features({
    'text': Value('string'),
    'image': DatasetImage(),
    'image_path': Value('string'),
    'filename': Value('string'),
    'class_id': Value('int64'),
    'character': Value('string'),
    'type': Value('string'),
    'augmentation': Value('string')
})

# Create dataset from generator
new_ds = Dataset.from_generator(generate_examples, features=features)
print(f"   ✅ Created dataset with {len(new_ds):,} samples")

# Step 4: Merge and create splits
print(f"\n[4/5] 🔗 Merging and creating splits...")

if existing_ds:
    print(f"   Merging {existing_count:,} + {len(new_ds):,} samples...")
    merged_ds = concatenate_datasets([existing_ds, new_ds])
    del existing_ds, new_ds
    gc.collect()
else:
    merged_ds = new_ds

total = len(merged_ds)
print(f"   ✅ Total: {total:,} samples")

# Shuffle
print(f"   Shuffling with seed {RANDOM_SEED}...")
merged_ds = merged_ds.shuffle(seed=RANDOM_SEED)

# Calculate splits
train_size = int(total * TRAIN_RATIO)
dev_size = int(total * DEV_RATIO)
test_size = total - train_size - dev_size

print(f"\n   Creating splits:")
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

print(f"\n   ✅ Splits created")

del merged_ds, train_ds, val_ds, test_ds
gc.collect()

# Step 5: Upload
print(f"\n[5/5] 📤 Uploading to HuggingFace...")
print(f"   Repository: {OUTPUT_DATASET_REPO}")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN not set")
    exit(1)

try:
    print(f"   ⏳ Uploading {total:,} samples...")
    dataset_dict.push_to_hub(
        OUTPUT_DATASET_REPO,
        token=hf_token,
        commit_message=f"Add {len(new_data):,} samples + create train/dev/test splits ({train_size}/{dev_size}/{test_size})"
    )
    
    print(f"\n✅ Success!")
    print(f"   🔗 https://huggingface.co/datasets/{OUTPUT_DATASET_REPO}")
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"   Total: {total:,} samples")
    print(f"\n   Splits:")
    print(f"      - train:      {len(dataset_dict['train']):,}")
    print(f"      - validation: {len(dataset_dict['validation']):,}")
    print(f"      - test:       {len(dataset_dict['test']):,}")
    print(f"\n   Usage:")
    print(f"      dataset = load_dataset('{OUTPUT_DATASET_REPO}')")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
