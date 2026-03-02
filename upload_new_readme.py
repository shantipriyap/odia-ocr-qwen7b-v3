#!/usr/bin/env python3
"""
Push the updated README to HuggingFace repository
"""

from huggingface_hub import upload_file
from pathlib import Path

model_id = "shantipriya/odia-ocr-qwen-finetuned"
readme_path = "/Users/shantipriya/work/odia_ocr/README_FINAL.md"

print("🚀 Uploading README to HuggingFace...")
print(f"Repository: {model_id}")
print(f"File: {readme_path}")

try:
    # Upload README
    commit_info = upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=model_id,
        repo_type="model",
        commit_message="Update README with comprehensive documentation and examples"
    )
    
    print("\n✅ README successfully uploaded!")
    print(f"Commit: {commit_info.commit_url}")
    print(f"\n📌 Changes include:")
    print("  - Author attribution (Shantipriya Parida)")
    print("  - Complete dataset documentation (58,720 samples)")
    print("  - Latest training metrics (Loss: 5.5 → 0.09)")
    print("  - 3 detailed long-form examples")
    print("  - Performance metrics and evaluation results")
    print("  - Installation and usage instructions")
    print("  - Future improvements roadmap")
    
    print(f"\n🌐 View at: https://huggingface.co/{model_id}")
    
except Exception as e:
    print(f"\n❌ Error uploading README: {e}")
    import traceback
    traceback.print_exc()
