from huggingface_hub import upload_file
from pathlib import Path

token = "os.getenv("HF_TOKEN", "")"
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

print("=" * 70)
print("🚀 UPDATING REQUIREMENTS.TXT")
print("=" * 70)

req_path = Path("/Users/shantipriya/work/odia_ocr/requirements.txt")

try:
    print("\n📤 Uploading updated requirements.txt...")
    print("   • Gradio: 4.26.0 → 6.6.0")
    print("   • Spaces: 0.28.0 → 0.47.0")
    
    upload_file(
        path_or_fileobj=str(req_path),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="🔧 Fix: Update Gradio to 6.6.0 - resolve dependency conflict"
    )
    print("\n✅ Updated successfully!")
    print("\n" + "=" * 70)
    print("🎉 Space is rebuilding...")
    print("=" * 70)
    print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
    print("⏳ Build in progress (2-3 minutes)")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {str(e)[:300]}")
    exit(1)
