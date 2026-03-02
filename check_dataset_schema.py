#!/usr/bin/env python3
"""
Check existing dataset schema
"""

from datasets import load_dataset

DATASET = "shantipriya/odia-ocr-merged"

print("📊 Checking existing dataset schema...")
try:
    dataset = load_dataset(DATASET, split="train")
    print(f"\n✅ Dataset loaded: {len(dataset)} samples")
    print(f"\n📋 Schema:")
    for field_name, field_type in dataset.features.items():
        print(f"   - {field_name}: {field_type}")
    
    print(f"\n📄 Sample data (first 3 rows):")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n   Sample {i+1}:")
        for key, value in sample.items():
            if key == 'image':
                print(f"      {key}: <PIL.Image>")
            else:
                val_str = str(value)[:50]
                print(f"      {key}: {val_str}")
                
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
