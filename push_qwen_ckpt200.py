import os
from huggingface_hub import HfApi

HF_TOKEN = "YOUR_HF_TOKEN_HERE"
REPO_ID  = "shantipriya/odia-ocr-qwen-finetuned_v3"
CKPT_DIR = "/root/phase3_paragraph/output_2gpu/checkpoint-200"

api = HfApi(token=HF_TOKEN)

print(f"Pushing {CKPT_DIR} to {REPO_ID} ...")
api.upload_folder(
    folder_path=CKPT_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload checkpoint-200 (step 200/3000)",
    ignore_patterns=["optimizer.pt", "scheduler.pt", "trainer_state.json",
                     "training_args.bin", "rng_state*.pth"],
)
print("✓ Checkpoint pushed")
