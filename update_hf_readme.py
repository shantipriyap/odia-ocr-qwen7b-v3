#!/usr/bin/env python3
"""
Update README on HuggingFace Hub
"""
import os
from huggingface_hub import HfApi

# Configuration
HF_REPO = "shantipriya/qwen2.5-odia-ocr-v2"
LOCAL_README = "./README.md"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN environment variable not set")
    exit(1)

print(f"""
📝 Updating README on HuggingFace Hub
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Repository: {HF_REPO}
Local File: {LOCAL_README}
""")

api = HfApi()

# Upload README
print("📤 Uploading README.md...")
try:
    api.upload_file(
        path_or_fileobj=LOCAL_README,
        path_in_repo="README.md",
        repo_id=HF_REPO,
        token=HF_TOKEN,
        repo_type="model",
        commit_message="Update README with Phase 2C completion (500/500 steps)"
    )
    print("✅ README updated successfully!")
    print(f"\n🔗 View at: https://huggingface.co/{HF_REPO}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    exit(1)
