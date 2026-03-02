#!/usr/bin/env python3
import os
from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

print("=" * 70)
print("🚀 DIRECT DEPLOYMENT TO HUGGINGFACE SPACE")
print("=" * 70)

# Get token
token = os.getenv("HF_TOKEN")
if not token:
    print("❌ Error: HF_TOKEN not set")
    print("Set it with: export HF_TOKEN='your_token'")
    exit(1)

api = HfApi(token=token)
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
repo_type = "space"

# Read files
app_path = Path("/Users/shantipriya/work/odia_ocr/app.py")
req_path = Path("/Users/shantipriya/work/odia_ocr/requirements.txt")

if not app_path.exists() or not req_path.exists():
    print(f"❌ Error: Missing files")
    print(f"   app.py: {app_path.exists()}")
    print(f"   requirements.txt: {req_path.exists()}")
    exit(1)

app_content = app_path.read_text()
req_content = req_path.read_text()

print(f"📝 Files to deploy:")
print(f"   • app.py ({len(app_content)} bytes)")
print(f"   • requirements.txt ({len(req_content)} bytes)")
print()

try:
    print("🔄 Uploading to Space...")
    api.create_commit(
        repo_id=space_id,
        repo_type=repo_type,
        operations=[
            CommitOperationAdd("app.py", app_content),
            CommitOperationAdd("requirements.txt", req_content),
        ],
        commit_message="🔧 Direct Deploy: Fixed Gradio app with all compatibility fixes",
    )
    
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print()
    print("=" * 70)
    print("🎉 Space deployed!")
    print("=" * 70)
    print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
    print("⏳ Space will rebuild and be live in 2-3 minutes")
    print("=" * 70)
    
except Exception as e:
    error_msg = str(e)
    print(f"❌ Deployment failed: {error_msg[:300]}")
    
    if "403" in error_msg or "Forbidden" in error_msg or "create_pr" in error_msg:
        print("\n⚠️  Token permission issue detected.")
        print("\nAlternatives:")
        print("1. Use PR merge: Check PR #2 at the Space and merge it")
        print("2. Grant permissions: Admin must add write access")
        print("3. Personal Space: Deploy to your own account instead")
    
    exit(1)
