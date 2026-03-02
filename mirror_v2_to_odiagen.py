# -*- coding: utf-8 -*-
"""
Mirror shantipriya/odia-ocr-qwen-finetuned_v2
       → OdiaGenAIOCR/odia-ocr-qwen-finetuned_v2
Copies all files, updates README adapter path.
"""
import io
import requests
from huggingface_hub import HfApi, list_repo_files, hf_hub_download

HF_TOKEN = "os.getenv("HF_TOKEN", "")"
SRC = "shantipriya/odia-ocr-qwen-finetuned_v2"
DST = "OdiaGenAIOCR/odia-ocr-qwen-finetuned_v2"

api = HfApi(token=HF_TOKEN)

# ── 1. Create destination repo if absent ─────────────────────────────────────
try:
    api.repo_info(repo_id=DST, repo_type="model")
    print(f"✅ Repo {DST} already exists")
except Exception:
    api.create_repo(repo_id=DST, repo_type="model", private=False, exist_ok=True)
    print(f"✅ Created repo {DST}")

# ── 2. Collect files to copy ──────────────────────────────────────────────────
all_files = sorted(list_repo_files(SRC, token=HF_TOKEN))

# Copy everything to make repos identical
SKIP = {".gitattributes"}
SKIP_SUFFIX = ()  # no skips — mirror all files

files_to_copy = [
    f for f in all_files
    if f not in SKIP and not any(f.endswith(s) for s in SKIP_SUFFIX)
]

print(f"\nFiles to mirror: {len(files_to_copy)}")
for f in files_to_copy:
    print(f"  {f}")

# ── 3. Copy file by file ──────────────────────────────────────────────────────
for path in files_to_copy:
    print(f"\n→ {path}", end="  ", flush=True)

    # Download from source
    local = hf_hub_download(repo_id=SRC, filename=path, token=HF_TOKEN)

    # Patch README: update model path references
    if path == "README.md":
        with open(local, "r", encoding="utf-8") as fh:
            content = fh.read()

        content = content.replace(
            "shantipriya/odia-ocr-qwen-finetuned_v2",
            "OdiaGenAIOCR/odia-ocr-qwen-finetuned_v2",
        )
        # Also patch the YAML front-matter model name
        content = content.replace(
            "- name: odia-ocr-qwen-finetuned_v2",
            "- name: OdiaGenAIOCR/odia-ocr-qwen-finetuned_v2",
        )
        # Update Contact mirror link
        content = content.replace(
            "- **Mirror**: [OdiaGenAIOCR/odia-ocr-qwen-finetuned](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned)",
            "- **Original**: [shantipriya/odia-ocr-qwen-finetuned_v2](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned_v2)",
        )

        data = content.encode("utf-8")
        api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=path,
            repo_id=DST,
            repo_type="model",
            commit_message=f"Mirror {path} (patched adapter path)",
        )
        print("✅ (patched)")
        continue

    # Upload binary as-is
    with open(local, "rb") as fh:
        api.upload_file(
            path_or_fileobj=fh,
            path_in_repo=path,
            repo_id=DST,
            repo_type="model",
            commit_message=f"Mirror {path}",
        )
    print("✅")

print(f"\n{'='*60}")
print(f"✅ Mirror complete!")
print(f"   https://huggingface.co/{DST}")
print(f"{'='*60}")
