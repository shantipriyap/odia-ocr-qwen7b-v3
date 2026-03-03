#!/usr/bin/env python3
"""
Extract the second image and text from OdiaGenAIOCR dataset
"""

from datasets import load_dataset
from PIL import Image

print("Loading OdiaGenAIOCR dataset...")
dataset = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data", split="train")

print(f"Loaded {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")

# Get second sample (index 1)
second_sample = dataset[1]

print("\n" + "="*70)
print("SECOND SAMPLE FROM DATASET")
print("="*70)

# Display the text
print(f"\nText:")
print(f"{second_sample['text']}")

# Display image info
image = second_sample['image']
print(f"\nImage Info:")
print(f"  Size: {image.size}")
print(f"  Mode: {image.mode}")

# Save the image
output_path = "/Users/shantipriya/work/odia_ocr/second_sample_image.png"
image.save(output_path)
print(f"\nImage saved to: {output_path}")

print("\n" + "="*70)
print("TEXT TO USE IN README:")
print("="*70)
print(second_sample['text'])
