#!/usr/bin/env python3
"""Direct upload to HF Space"""

from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

print("=" * 70)
print("📤 UPLOADING FILES DIRECTLY...")
print("=" * 70)

try:
    operations = [
        CommitOperationAdd(
            path_in_repo="app.py",
            path_or_fileobj="/Users/shantipriya/work/odia_ocr/app.py"
        ),
        CommitOperationAdd(
            path_in_repo="requirements.txt",
            path_or_fileobj="/Users/shantipriya/work/odia_ocr/requirements.txt"
        ),
    ]
    
    commit_info = api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operations,
        commit_message="🔧 Fix: Remove unsupported show_copy_button parameter"
    )
    
    print("\n✅ DEPLOYED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Commit: {commit_info.commit_url}")
    print(f"Space: https://huggingface.co/spaces/{space_id}")
    print("⏳ Building... (2-3 minutes)")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
