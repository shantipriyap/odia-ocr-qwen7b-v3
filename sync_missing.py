from huggingface_hub import HfApi, hf_hub_download, list_repo_files

token = "os.getenv("HF_TOKEN", "")"
SRC = "shantipriya/odia-ocr-qwen-finetuned_v2"
DST = "OdiaGenAIOCR/odia-ocr-qwen-finetuned_v2"
api = HfApi(token=token)

missing = [
    "checkpoint-5000/optimizer.pt",
    "checkpoint-5000/rng_state.pth",
    "checkpoint-5000/scheduler.pt",
    "checkpoint-5400/optimizer.pt",
    "checkpoint-5400/rng_state.pth",
    "checkpoint-5400/scheduler.pt",
    "checkpoint-6000/optimizer.pt",
    "checkpoint-6000/rng_state.pth",
    "checkpoint-6000/scheduler.pt",
]

for path in missing:
    print(f"-> {path}", end="  ", flush=True)
    local = hf_hub_download(repo_id=SRC, filename=path, token=token)
    with open(local, "rb") as fh:
        api.upload_file(
            path_or_fileobj=fh,
            path_in_repo=path,
            repo_id=DST,
            repo_type="model",
            commit_message=f"Sync {path}",
        )
    print("OK")

src = set(list_repo_files(SRC, token=token))
dst = set(list_repo_files(DST, token=token))
diff = src - dst
print("\n=== RESULT ===")
print(f"shantipriya: {len(src)}  OdiaGenAIOCR: {len(dst)}")
if not diff:
    print("IDENTICAL - both repos are fully in sync")
else:
    print(f"Still missing {len(diff)}: {diff}")
