#!/usr/bin/env python3
"""Upload app.py to HuggingFace Space"""

from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

print("=" * 70)
print("📤 UPLOADING TO HUGGINGFACE SPACE")
print("=" * 70)

try:
    operations = [
        CommitOperationAdd(path_in_repo="app.py", path_or_fileobj="/Users/shantipriya/work/odia_ocr/app.py"),
        CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj="/Users/shantipriya/work/odia_ocr/requirements.txt"),
    ]
    
    print("📝 app.py (8.3 KB)")
    print("📝 requirements.txt (226 B)")
    
    commit_info = api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operations,
        commit_message="🎨 Deploy Gradio-based OCR with @spaces.GPU decorator"
    )
    
    print("\n" + "=" * 70)
    print("✅ UPLOAD SUCCESSFUL!")
    print("=" * 70)
    print(f"Space: https://huggingface.co/spaces/{space_id}")
    print("⏳ Building... (2-3 minutes)")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
