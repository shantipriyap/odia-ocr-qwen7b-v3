from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

token = "os.getenv("HF_TOKEN", "")"
api = HfApi(token=token)
space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"

print("=" * 70)
print("🚀 FORCE UPDATE - GRADIO FIX")
print("=" * 70)

req_path = Path("/Users/shantipriya/work/odia_ocr/requirements.txt")
req_content = req_path.read_text()

print(f"\n📊 Content preview:")
print(req_content[:200] + "...")

try:
    print("\n🔄 Force updating requirements.txt...")
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=[
            CommitOperationAdd("requirements.txt", req_content),
        ],
        commit_message="🔧 FORCE FIX: Gradio 6.6.0 - resolve dependency conflict"
    )
    
    print("\n✅ Force update successful!")
    print("=" * 70)
    print("🎉 Space rebuilding with Gradio 6.6.0")
    print("=" * 70)
    print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
    print("⏳ Build in progress (2-3 minutes)")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {str(e)[:300]}")
    exit(1)
