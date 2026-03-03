#!/usr/bin/env python3
"""
Upload checkpoint-500 to HuggingFace Hub
"""
import os
from huggingface_hub import HfApi

# Configuration
HF_REPO = "shantipriya/qwen2.5-odia-ocr-v2"
LOCAL_CHECKPOINT = "./checkpoint-500-phase2c"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN environment variable not set")
    print("Set it with: export HF_TOKEN='your_token_here'")
    exit(1)

print(f"""
🚀 Uploading Checkpoint-500 to HuggingFace Hub
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Repository: {HF_REPO}
Local Path: {LOCAL_CHECKPOINT}
""")

api = HfApi()

# Upload checkpoint folder
print("📤 Uploading checkpoint-500 files...")
try:
    api.upload_folder(
        folder_path=LOCAL_CHECKPOINT,
        repo_id=HF_REPO,
        path_in_repo="checkpoint-500",
        token=HF_TOKEN,
        repo_type="model"
    )
    print("✅ Upload complete!")
    print(f"\n🔗 View at: https://huggingface.co/{HF_REPO}/tree/main/checkpoint-500")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    exit(1)
