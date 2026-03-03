#!/usr/bin/env python3
"""
Merge odia dataset with existing HuggingFace dataset
Verifies fields and creates consistent schema
"""

import os
from datasets import load_dataset, Dataset, Features, Value, Image as DatasetImage, concatenate_datasets
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Configuration
EXISTING_DATASET = "shantipriya/odia-ocr-merged"
NEW_DATA_PATH = "/Users/shantipriya/work/odia_ocr/odia"
LABELS_FILE = f"{NEW_DATA_PATH}/train.txt"
OUTPUT_DATASET_REPO = "shantipriya/odia-ocr-merged"

print("="*80)
print("📊 ODIA DATASET MERGE & VERIFICATION")
print("="*80)

# Step 1: Load existing dataset
print("\n[1/5] 📥 Loading existing dataset from HuggingFace...")
try:
    existing_dataset = load_dataset(EXISTING_DATASET, split="train")
    print(f"✅ Loaded {len(existing_dataset)} existing samples")
    print(f"   Schema: {existing_dataset.features}")
except Exception as e:
    print(f"⚠️  Could not load existing dataset: {e}")
    existing_dataset = None

# Step 2: Verify new dataset
print(f"\n[2/5] 🔍 Verifying new odia dataset...")
with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Total samples: {len(lines)}")
print(f"   Sample format check:")

# Check first few samples
for i, line in enumerate(lines[:3]):
    parts = line.strip().split(' ', 1)
    if len(parts) == 2:
        img_path, text = parts
        full_path = f"{NEW_DATA_PATH}/{img_path}"
        exists = "✅" if os.path.exists(full_path) else "❌"
        print(f"      {i+1}. {exists} {img_path} → '{text[:30]}...'")

# Step 3: Process new dataset with consistent schema
print(f"\n[3/5] 🔄 Processing new dataset with consistent schema...")

# Define schema matching existing dataset
features = Features({
    'text': Value('string'),
    'image': DatasetImage(),
    'image_path': Value('string'),
    'filename': Value('string'),
    'class_id': Value('string'),
    'character': Value('string'),
    'type': Value('string'),
    'augmentation': Value('string')
})

data = {
    'text': [],
    'image': [],
    'image_path': [],
    'filename': [],
    'class_id': [],
    'character': [],
    'type': [],
    'augmentation': []
}

print(f"   Processing {len(lines)} samples...")
processed = 0
skipped = 0

for line in tqdm(lines, desc="Processing"):
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
        # Load image
        image = Image.open(full_path)
        
        # Add to dataset with consistent schema
        data['text'].append(text)
        data['image'].append(image)
        data['image_path'].append(img_path)
        data['filename'].append(os.path.basename(img_path))
        data['class_id'].append("None")
        data['character'].append("None")
        data['type'].append("word")  # Since these appear to be word-level OCR
        data['augmentation'].append("None")
        
        processed += 1
        
    except Exception as e:
        print(f"\n⚠️  Error processing {img_path}: {e}")
        skipped += 1

print(f"\n   ✅ Processed: {processed} samples")
print(f"   ⚠️  Skipped: {skipped} samples")

# Create new dataset
new_dataset = Dataset.from_dict(data, features=features)
print(f"\n   Schema verification:")
print(f"      New dataset: {new_dataset.features}")
if existing_dataset:
    print(f"      Existing dataset: {existing_dataset.features}")
    schema_match = new_dataset.features == existing_dataset.features
    print(f"      Schemas match: {'✅' if schema_match else '❌'}")

# Step 4: Merge datasets
print(f"\n[4/5] 🔗 Merging datasets...")
if existing_dataset:
    merged_dataset = concatenate_datasets([existing_dataset, new_dataset])
    print(f"   ✅ Merged dataset size: {len(merged_dataset)} samples")
    print(f"      - Existing: {len(existing_dataset)}")
    print(f"      - New: {len(new_dataset)}")
    print(f"      - Total: {len(merged_dataset)}")
else:
    merged_dataset = new_dataset
    print(f"   ✅ Using new dataset only: {len(merged_dataset)} samples")

# Step 5: Upload to HuggingFace
print(f"\n[5/5] 📤 Uploading merged dataset to HuggingFace...")
print(f"   Repository: {OUTPUT_DATASET_REPO}")
print(f"   ⏳ This may take several minutes...")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN environment variable not set")
    exit(1)

try:
    merged_dataset.push_to_hub(
        OUTPUT_DATASET_REPO,
        token=hf_token,
        commit_message=f"Add {processed} new Odia OCR samples from odia dataset"
    )
    print(f"\n✅ Upload complete!")
    print(f"   View at: https://huggingface.co/datasets/{OUTPUT_DATASET_REPO}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 MERGE SUMMARY")
    print("="*80)
    if existing_dataset:
        print(f"   Original dataset size: {len(existing_dataset):,}")
    print(f"   New samples added: {processed:,}")
    print(f"   Final dataset size: {len(merged_dataset):,}")
    print(f"   Samples skipped: {skipped:,}")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error uploading: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
