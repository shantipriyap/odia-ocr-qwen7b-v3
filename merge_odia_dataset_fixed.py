#!/usr/bin/env python3
"""
Merge odia dataset with existing HuggingFace dataset
Properly matches schema and handles file descriptors
"""

import os
from datasets import load_dataset, Dataset, Features, Value, Image as DatasetImage, concatenate_datasets
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc

# Configuration
EXISTING_DATASET = "shantipriya/odia-ocr-merged"
NEW_DATA_PATH = "/Users/shantipriya/work/odia_ocr/odia"
LABELS_FILE = f"{NEW_DATA_PATH}/train.txt"
OUTPUT_DATASET_REPO = "shantipriya/odia-ocr-merged"
BATCH_SIZE = 1000  # Process in batches to avoid file descriptor issues

print("="*80)
print("📊 ODIA DATASET MERGE & VERIFICATION")
print("="*80)

# Step 1: Load existing dataset
print("\n[1/5] 📥 Loading existing dataset from HuggingFace...")
try:
    existing_dataset = load_dataset(EXISTING_DATASET, split="train")
    print(f"✅ Loaded {len(existing_dataset)} existing samples")
    print(f"\n   Schema:")
    for field_name, field_type in existing_dataset.features.items():
        print(f"      - {field_name}: {field_type}")
except Exception as e:
    print(f"⚠️  Could not load existing dataset: {e}")
    existing_dataset = None

# Step 2: Read and verify labels
print(f"\n[2/5] 🔍 Verifying new odia dataset...")
with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Total samples in labels file: {len(lines)}")

# Verify first few samples
print(f"\n   Verifying first 3 samples:")
for i, line in enumerate(lines[:3]):
    parts = line.strip().split(' ', 1)
    if len(parts) == 2:
        img_path, text = parts
        full_path = f"{NEW_DATA_PATH}/{img_path}"
        exists = "✅" if os.path.exists(full_path) else "❌"
        print(f"      {i+1}. {exists} {img_path} → '{text[:30]}...'")

# Step 3: Define schema matching existing dataset (with int64 for class_id)
print(f"\n[3/5] 🔄 Processing new dataset with matching schema...")

features = Features({
    'text': Value('string'),
    'image': DatasetImage(),
    'image_path': Value('string'),
    'filename': Value('string'),
    'class_id': Value('int64'),  # Must be int64, not string!
    'character': Value('string'),
    'type': Value('string'),
    'augmentation': Value('string')
})

# Process in batches to avoid "too many open files" error
all_data = {
    'text': [],
    'image': [],
    'image_path': [],
    'filename': [],
    'class_id': [],
    'character': [],
    'type': [],
    'augmentation': []
}

processed = 0
skipped = 0

print(f"   Processing {len(lines)} samples in batches of {BATCH_SIZE}...")

for batch_start in range(0, len(lines), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(lines))
    batch_lines = lines[batch_start:batch_end]
    
    print(f"\n   Batch {batch_start//BATCH_SIZE + 1}/{(len(lines)-1)//BATCH_SIZE + 1} " + 
          f"(samples {batch_start+1}-{batch_end})...")
    
    batch_images = []
    batch_data = []
    
    for line in tqdm(batch_lines, desc="   Loading"):
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
            # Load image and immediately convert to bytes to avoid keeping file open
            with Image.open(full_path) as img:
                img_copy = img.copy()  # Make a copy to close file
            
            batch_data.append({
                'text': text,
                'image': img_copy,
                'image_path': img_path,
                'filename': os.path.basename(img_path),
                'class_id': None,  # None gets converted to proper null in int64
                'character': None,
                'type': "word",
                'augmentation': None
            })
            
            processed += 1
            
        except Exception as e:
            print(f"\n      ⚠️  Error processing {img_path}: {e}")
            skipped += 1
    
    # Add batch to main dataset
    for item in batch_data:
        for key in all_data.keys():
            all_data[key].append(item[key])
    
    # Clean up batch
    del batch_images
    del batch_data
    gc.collect()
    
    print(f"      Processed: {processed}, Skipped: {skipped}")

print(f"\n   ✅ Total processed: {processed} samples")
print(f"   ⚠️  Total skipped: {skipped} samples")

# Create new dataset
print(f"\n   Creating dataset from processed samples...")
new_dataset = Dataset.from_dict(all_data, features=features)

print(f"\n   Schema verification:")
print(f"      New dataset schema:")
for field_name, field_type in new_dataset.features.items():
    print(f"         - {field_name}: {field_type}")

if existing_dataset:
    print(f"\n      Existing dataset schema:")
    for field_name, field_type in existing_dataset.features.items():
        print(f"         - {field_name}: {field_type}")
    
    schema_match = new_dataset.features == existing_dataset.features
    print(f"\n      Schemas match: {'✅ Yes' if schema_match else '❌ No'}")

# Step 4: Merge datasets
print(f"\n[4/5] 🔗 Merging datasets...")
if existing_dataset:
    print(f"   Merging {len(existing_dataset)} existing + {len(new_dataset)} new samples...")
    merged_dataset = concatenate_datasets([existing_dataset, new_dataset])
    print(f"   ✅ Merged dataset: {len(merged_dataset)} total samples")
    print(f"      - Existing: {len(existing_dataset):,}")
    print(f"      - New: {len(new_dataset):,}")
    print(f"      - Total: {len(merged_dataset):,}")
else:
    merged_dataset = new_dataset
    print(f"   ✅ Using new dataset only: {len(merged_dataset)} samples")

# Step 5: Upload to HuggingFace
print(f"\n[5/5] 📤 Uploading merged dataset to HuggingFace...")
print(f"   Repository: {OUTPUT_DATASET_REPO}")
print(f"   ⏳ This may take several minutes for {len(merged_dataset)} samples...")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN environment variable not set")
    exit(1)

try:
    merged_dataset.push_to_hub(
        OUTPUT_DATASET_REPO,
        token=hf_token,
        commit_message=f"Add {processed:,} new Odia OCR word-level samples (total: {len(merged_dataset):,})"
    )
    print(f"\n✅ Upload complete!")
    print(f"   View at: https://huggingface.co/datasets/{OUTPUT_DATASET_REPO}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 MERGE SUMMARY")
    print("="*80)
    if existing_dataset:
        print(f"   Original dataset size: {len(existing_dataset):,} samples")
    print(f"   New samples added: {processed:,} samples")
    print(f"   Final dataset size: {len(merged_dataset):,} samples")
    print(f"   Samples skipped: {skipped:,} samples")
    print(f"\n   ✅ All samples now follow consistent schema:")
    print(f"      - text: Odia text content")
    print(f"      - image: PIL Image")
    print(f"      - image_path: File path reference")
    print(f"      - filename: Image filename")
    print(f"      - class_id: Integer class ID (int64)")
    print(f"      - character: Character label")
    print(f"      - type: Sample type (word/character)")
    print(f"      - augmentation: Augmentation applied")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error uploading: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
