#!/usr/bin/env python3
"""
Upload odia.zip to HuggingFace dataset repository
"""

import os
from huggingface_hub import HfApi
from pathlib import Path

# Configuration
HF_DATASET_REPO = "shantipriya/odia-ocr-merged"
ZIP_FILE_PATH = str(Path.home() / "Downloads" / "odia.zip")

# Get HF token from environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN environment variable not set")
    print("   Run: export HF_TOKEN='your_token_here'")
    exit(1)

# Check if file exists
if not os.path.exists(ZIP_FILE_PATH):
    print(f"❌ File not found: {ZIP_FILE_PATH}")
    exit(1)

file_size_mb = os.path.getsize(ZIP_FILE_PATH) / (1024 * 1024)
print(f"📦 Uploading file to HuggingFace dataset")
print(f"   Dataset: {HF_DATASET_REPO}")
print(f"   File: {ZIP_FILE_PATH}")
print(f"   Size: {file_size_mb:.1f} MB")
print(f"\n⏳ This may take several minutes for a large file...")

# Initialize API
api = HfApi()

try:
    # Upload file to dataset repository
    api.upload_file(
        path_or_fileobj=ZIP_FILE_PATH,
        path_in_repo="odia.zip",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=hf_token,
        commit_message="Add odia.zip dataset"
    )
    
    print(f"\n✅ File uploaded successfully!")
    print(f"   View at: https://huggingface.co/datasets/{HF_DATASET_REPO}/tree/main")
    
except Exception as e:
    print(f"\n❌ Error uploading file: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
