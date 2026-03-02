import os
import glob
import pyarrow.parquet as pq

# Directory containing the downloaded parquet files
DATA_DIR = "/root/.cache/huggingface/hub/datasets--OdiaGenAIdata--Reasoning_GU/snapshots/e7b6b8082cbd43fef4f7b6f4a8946985ea3ba0f8/data"

# Find all parquet files
parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))

if not parquet_files:
    print("No parquet files found in data directory.")
    exit(1)

print(f"Found {len(parquet_files)} parquet files.")

# Print number of rows in each file
for f in parquet_files:
    try:
        table = pq.read_table(f)
        num_rows = table.num_rows
        print(f"{os.path.basename(f)}: {num_rows} rows")
    except Exception as e:
        print(f"Error reading {f}: {e}")
