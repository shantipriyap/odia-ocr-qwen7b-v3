#!/usr/bin/env python3
"""
Update HuggingFace model repository README with evaluation results
"""

import os
from huggingface_hub import HfApi

# Configuration
HF_REPO = "shantipriya/qwen2.5-odia-ocr-v2"
README_PATH = "/Users/shantipriya/work/odia_ocr/README.md"

# Get HF token from environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ HF_TOKEN environment variable not set")
    exit(1)

print(f"📤 Uploading README to HuggingFace: {HF_REPO}")

# Initialize API
api = HfApi()

try:
    # Upload README
    api.upload_file(
        path_or_fileobj=README_PATH,
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
        token=hf_token,
        commit_message="Update README with checkpoint-500 evaluation results (Telugu script issue detected)"
    )
    
    print(f"✅ README updated successfully")
    print(f"   View at: https://huggingface.co/{HF_REPO}")
    
except Exception as e:
    print(f"❌ Error uploading README: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
