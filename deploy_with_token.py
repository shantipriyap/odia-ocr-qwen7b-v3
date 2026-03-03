#!/usr/bin/env python3
from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

token = "os.getenv("HF_TOKEN", "")"
api = HfApi(token=token)

print("=" * 70)
print("🚀 DIRECT DEPLOYMENT TO HUGGINGFACE SPACE")
print("=" * 70)

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
print(f"   • app.py ({len(app_content):,} bytes)")
print(f"   • requirements.txt ({len(req_content):,} bytes)")
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
        commit_message="🔧 Deploy: Fixed Gradio app - direct production deployment",
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
    print(f"❌ Deployment failed!")
    print(f"\nError: {error_msg[:500]}")
    print("\nDebugging info:")
    print(f"  Token (masked): hf_***{token[-10:]}")
    print(f"  Space ID: {space_id}")
    exit(1)
