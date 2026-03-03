import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import Dataset
from huggingface_hub import snapshot_download, HfApi

# =========================
# CONFIG
# =========================
SOURCE_DATASET = "OdiaGenAIdata/Reasoning_GU"
TARGET_DATASET = "OdiaGenAIdata/Reasoning_GU"  # same repo
DATA_DIR = "data"
OUTPUT_FILE = "train.parquet"
BRANCH = "main"

# =========================
# STEP 1: Download dataset
# =========================
local_dir = snapshot_download(
    repo_id=SOURCE_DATASET,
    repo_type="dataset",
    revision=BRANCH
)

parquet_files = glob.glob(
    os.path.join(local_dir, DATA_DIR, "*.parquet")
)

print(f"Found {len(parquet_files)} parquet shards")

# =========================
# STEP 2: Merge parquet files
# =========================
dfs = []
for f in parquet_files:
    df = pq.read_table(f).to_pandas()
    if len(df) > 0:
        dfs.append(df)

if not dfs:
    raise ValueError("❌ All parquet shards are empty")

merged_df = pd.concat(dfs, ignore_index=True)
print("Merged shape:", merged_df.shape)

# =========================
# STEP 3: Save as single parquet
# =========================
table = pa.Table.from_pandas(merged_df)
pq.write_table(table, OUTPUT_FILE)

print(f"Saved merged file -> {OUTPUT_FILE}")

# =========================
# STEP 4: Push to Hugging Face
# =========================
api = HfApi()

api.upload_file(
    path_or_fileobj=OUTPUT_FILE,
    path_in_repo="data/train.parquet",
    repo_id=TARGET_DATASET,
    repo_type="dataset",
    commit_message="Add merged train.parquet for dataset viewer"
)

print("✅ Uploaded train.parquet successfully")
