from huggingface_hub import list_repo_files
import os

repo_id = "OdiaGenAIdata/Reasoning_GU"

print(f"Listing files in Hugging Face dataset repo: {repo_id}\n")
files = list_repo_files(repo_id, repo_type="dataset")
for f in files:
    print(f)

print("\nYou can download and inspect any of these files directly using the URLs:")
print(f"https://huggingface.co/datasets/{repo_id}/resolve/main/<filename>")

# Example: Download and inspect a parquet file if found
parquet_files = [f for f in files if f.endswith('.parquet')]
if parquet_files:
    print("\nSample parquet file found:", parquet_files[0])
    print(f"Direct URL: https://huggingface.co/datasets/{repo_id}/resolve/main/{parquet_files[0]}")
    try:
        import pandas as pd
        import requests
        import io
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{parquet_files[0]}"
        r = requests.get(url)
        df = pd.read_parquet(io.BytesIO(r.content))
        print("\nFirst 5 rows of the parquet file:")
        print(df.head())
    except Exception as e:
        print(f"[ERROR] Could not read parquet file: {e}")
else:
    print("No parquet files found in the repo.")
