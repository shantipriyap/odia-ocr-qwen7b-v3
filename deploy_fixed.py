#!/usr/bin/env python3
from huggingface_hub import upload_file
import os

token = os.environ.get('HF_TOKEN', 'os.getenv("HF_TOKEN", "")')
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

files_to_upload = [
    ("requirements.txt", "requirements.txt"),
    ("app.py", "app.py")
]

print("=" * 70)
print("📤 DEPLOYING FIXED VERSION")
print("=" * 70)

for local_path, remote_path in files_to_upload:
    print(f"\n📦 Uploading {local_path}...")
    try:
        upload_file(
            path_or_fileobj=f"/Users/shantipriya/work/odia_ocr/{local_path}",
            path_in_repo=remote_path,
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="🔧 Fix: Remove conflicting gradio - let Space manage"
        )
        print(f"✅ {local_path} uploaded")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("✅ DEPLOYMENT COMPLETE - Space rebuilding")
print("=" * 70)
print(f"Monitor: https://huggingface.co/spaces/{space_id}")
