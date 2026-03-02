#!/usr/bin/env python3
"""
Memory-efficient merge of odia dataset with train/dev/test splits
"""

import os
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image as DatasetImage
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc
import random

# Configuration
EXISTING_DATASET = "shantipriya/odia-ocr-merged"
NEW_DATA_PATH = "/Users/shantipriya/work/odia_ocr/odia"
LABELS_FILE = f"{NEW_DATA_PATH}/train.txt"
OUTPUT_DATASET_REPO = "shantipriya/odia-ocr-merged"
CHUNK_SIZE = 500  # Smaller chunks for memory efficiency
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

print("="*80)
print("📊 ODIA DATASET MERGE & SPLIT")
print("="*80)

# Step 1: Load existing dataset
print("\n[1/6] 📥 Loading existing dataset from HuggingFace...")
try:
    existing_dataset = load_dataset(EXISTING_DATASET, split="train")
    print(f"✅ Loaded {len(existing_dataset)} existing samples")
except Exception as e:
    print(f"⚠️  Could not load existing dataset: {e}")
    existing_dataset = None

# Step 2: Process new data in chunks and save to disk
print(f"\n[2/6] 🔄 Processing new dataset in memory-efficient chunks...")

with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Total samples to process: {len(lines)}")

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

# Process and save in chunks to avoid memory overflow
chunk_datasets = []
processed = 0
skipped = 0

for chunk_start in range(0, len(lines), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(lines))
    chunk_lines = lines[chunk_start:chunk_end]
    
    chunk_num = chunk_start // CHUNK_SIZE + 1
    total_chunks = (len(lines) - 1) // CHUNK_SIZE + 1
    
    print(f"\n   Chunk {chunk_num}/{total_chunks} (samples {chunk_start+1}-{chunk_end})...")
    
    chunk_data = {
        'text': [],
        'image': [],
        'image_path': [],
        'filename': [],
        'class_id': [],
        'character': [],
        'type': [],
        'augmentation': []
    }
    
    for line in tqdm(chunk_lines, desc="   Processing", leave=False):
        parts = line.strip().split(' ', 1)
        if len(parts) != 2:
            skipped += 1
            continue
        
        img_path, text = parts
        full_path = f"{NEW_DATA_PATH}/{img_path}"
        
        if not os.path.exists(full_path):
            skipped += 1
            continue
        
        try:
            # Load image, convert to RGB, and close file immediately
            with Image.open(full_path) as img:
                # Convert to RGB to ensure consistency
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_copy = img.copy()
            
            chunk_data['text'].append(text)
            chunk_data['image'].append(img_copy)
            chunk_data['image_path'].append(img_path)
            chunk_data['filename'].append(os.path.basename(img_path))
            chunk_data['class_id'].append(None)
            chunk_data['character'].append(None)
            chunk_data['type'].append("word")
            chunk_data['augmentation'].append(None)
            
            processed += 1
            
        except Exception as e:
            if chunk_start < 5000:  # Only print errors for first few chunks
                print(f"\n      ⚠️  Error: {img_path}: {e}")
            skipped += 1
    
    # Create dataset from chunk
    if len(chunk_data['text']) > 0:
        chunk_dataset = Dataset.from_dict(chunk_data, features=features)
        chunk_datasets.append(chunk_dataset)
        print(f"      ✅ Chunk {chunk_num}: {len(chunk_dataset)} samples")
    
    # Clean up
    del chunk_data
    gc.collect()

print(f"\n   ✅ Processed: {processed:,} samples in {len(chunk_datasets)} chunks")
print(f"   ⚠️  Skipped: {skipped:,} samples")

# Step 3: Concatenate chunks
print(f"\n[3/6] 🔗 Concatenating chunks...")
from datasets import concatenate_datasets
new_dataset = concatenate_datasets(chunk_datasets)
print(f"   ✅ New dataset: {len(new_dataset):,} samples")

# Clean up chunks
del chunk_datasets
gc.collect()

# Step 4: Merge with existing dataset
print(f"\n[4/6] 🔗 Merging with existing dataset...")
if existing_dataset:
    print(f"   Merging {len(existing_dataset):,} existing + {len(new_dataset):,} new...")
    merged_dataset = concatenate_datasets([existing_dataset, new_dataset])
    print(f"   ✅ Total: {len(merged_dataset):,} samples")
    del existing_dataset, new_dataset
    gc.collect()
else:
    merged_dataset = new_dataset
    print(f"   ✅ Using new dataset only: {len(merged_dataset):,} samples")

# Step 5: Create train/dev/test splits
print(f"\n[5/6] ✂️  Creating train/dev/test splits...")
print(f"   Split ratios: Train={TRAIN_RATIO}, Dev={DEV_RATIO}, Test={TEST_RATIO}")

# Shuffle the dataset
merged_dataset = merged_dataset.shuffle(seed=RANDOM_SEED)

# Calculate split sizes
total_size = len(merged_dataset)
train_size = int(total_size * TRAIN_RATIO)
dev_size = int(total_size * DEV_RATIO)
test_size = total_size - train_size - dev_size

print(f"\n   Split sizes:")
print(f"      - Train: {train_size:,} samples ({train_size/total_size*100:.1f}%)")
print(f"      - Dev:   {dev_size:,} samples ({dev_size/total_size*100:.1f}%)")
print(f"      - Test:  {test_size:,} samples ({test_size/total_size*100:.1f}%)")

# Create splits
train_dataset = merged_dataset.select(range(train_size))
dev_dataset = merged_dataset.select(range(train_size, train_size + dev_size))
test_dataset = merged_dataset.select(range(train_size + dev_size, total_size))

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': dev_dataset,
    'test': test_dataset
})

print(f"\n   ✅ Splits created:")
print(f"      - train: {len(dataset_dict['train']):,}")
print(f"      - validation: {len(dataset_dict['validation']):,}")
print(f"      - test: {len(dataset_dict['test']):,}")

# Clean up
del merged_dataset, train_dataset, dev_dataset, test_dataset
gc.collect()

# Step 6: Upload to HuggingFace
print(f"\n[6/6] 📤 Uploading to HuggingFace...")
print(f"   Repository: {OUTPUT_DATASET_REPO}")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN environment variable not set")
    exit(1)

try:
    print(f"   ⏳ Uploading (this may take several minutes)...")
    dataset_dict.push_to_hub(
        OUTPUT_DATASET_REPO,
        token=hf_token,
        commit_message=f"Merge {processed:,} new samples + create train/dev/test splits (total: {total_size:,})"
    )
    
    print(f"\n✅ Upload complete!")
    print(f"   View at: https://huggingface.co/datasets/{OUTPUT_DATASET_REPO}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"   Total dataset size: {total_size:,} samples")
    print(f"\n   Splits:")
    print(f"      - train:      {len(dataset_dict['train']):,} ({len(dataset_dict['train'])/total_size*100:.1f}%)")
    print(f"      - validation: {len(dataset_dict['validation']):,} ({len(dataset_dict['validation'])/total_size*100:.1f}%)")
    print(f"      - test:       {len(dataset_dict['test']):,} ({len(dataset_dict['test'])/total_size*100:.1f}%)")
    print(f"\n   ✅ Dataset ready for training!")
    print(f"   Usage:")
    print(f"      from datasets import load_dataset")
    print(f"      dataset = load_dataset('{OUTPUT_DATASET_REPO}')")
    print(f"      # Access splits: dataset['train'], dataset['validation'], dataset['test']")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error uploading: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
