from huggingface_hub import upload_file
from pathlib import Path

token = "os.getenv("HF_TOKEN", "")"
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

print("=" * 70)
print("🚀 FIXING BUILD ERROR")
print("=" * 70)

req_path = Path("/Users/shantipriya/work/odia_ocr/requirements.txt")

try:
    print("\n📤 Deploying stable Gradio version...")
    print("   Gradio: 6.6.0 → 4.41.0 (stable)")
    print("   Spaces: 0.47.0 → 0.30.0 (stable)")
    
    upload_file(
        path_or_fileobj=str(req_path),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="🔧 Fix: Use stable Gradio 4.41.0 - resolve build error"
    )
    
    print("\n✅ Deployed successfully!")
    print("\n" + "=" * 70)
    print("🎉 Space rebuilding with stable versions...")
    print("=" * 70)
    print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
    print("⏳ Build in progress (2-3 minutes)")
    print("=" * 70)
    
except Exception as e:
    error_str = str(e)
    print(f"\n❌ Error: {error_str[:300]}")
    exit(1)
