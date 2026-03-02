import os
from huggingface_hub import snapshot_download

# =========================
# CONFIG
# =========================
SOURCE_DATASET = "OdiaGenAIdata/Reasoning_GU"
BRANCH = "main"

# =========================
# STEP 1: Download dataset
# =========================
local_dir = snapshot_download(
    repo_id=SOURCE_DATASET,
    repo_type="dataset",
    revision=BRANCH
)

print(f"Downloaded snapshot to: {local_dir}")

# List all files and directories recursively
for root, dirs, files in os.walk(local_dir):
    level = root.replace(local_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")
