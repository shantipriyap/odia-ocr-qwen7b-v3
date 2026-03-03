#!/usr/bin/env python3
from huggingface_hub import upload_file
from pathlib import Path

token = "os.getenv("HF_TOKEN", "")"

print("=" * 70)
print("🚀 DEPLOYING TO HUGGINGFACE SPACE")
print("=" * 70)

space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

app_path = Path("/Users/shantipriya/work/odia_ocr/app.py")
req_path = Path("/Users/shantipriya/work/odia_ocr/requirements.txt")

try:
    print("\n📤 Uploading app.py...")
    upload_file(
        path_or_fileobj=str(app_path),
        path_in_repo="app.py",
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="🔧 Deploy: Fixed Gradio app"
    )
    print("✅ app.py uploaded")
    
    print("\n📤 Uploading requirements.txt...")
    upload_file(
        path_or_fileobj=str(req_path),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="🔧 Deploy: Updated dependencies"
    )
    print("✅ requirements.txt uploaded")
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("=" * 70)
    print(f"🔗 Space: https://huggingface.co/spaces/{space_id}")
    print("⏳ Building... (check back in 2-3 minutes)")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {str(e)[:300]}")
    exit(1)
