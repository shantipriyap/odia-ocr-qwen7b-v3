from huggingface_hub import upload_file
from pathlib import Path

token = "os.getenv("HF_TOKEN", "")"
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

print("=" * 70)
print("🚀 UPLOADING FIXED REQUIREMENTS")
print("=" * 70)

req_path = Path("/Users/shantipriya/work/odia_ocr/requirements.txt")

try:
    print("\n📤 Uploading requirements.txt with Gradio 6.6.0...")
    
    result = upload_file(
        path_or_fileobj=str(req_path),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="🔧 Fix: Gradio 6.6.0 - resolve version conflict"
    )
    
    print("\n✅ Upload successful!")
    print(f"   Commit: {result}")
    print("\n" + "=" * 70)
    print("🎉 Space rebuilding...")
    print("=" * 70)
    print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
    print("⏳ Give it 2-3 minutes to build")
    print("=" * 70)
    
except Exception as e:
    error_str = str(e)
    print(f"\n❌ Error: {error_str[:300]}")
    if "No files have been modified" in error_str:
        print("\n✅ File already at latest version (no change needed)")
    exit(1)
