from datasets import get_dataset_config_names, load_dataset

repo = "OdiaGenAIdata/Reasoning_GU"

# List all splits (batches)
splits = get_dataset_config_names(repo)
print("Available splits:", splits)

# Try to load and print the first row from each split
for split in splits:
    print(f"\n--- Split: {split} ---")
    try:
        ds = load_dataset(repo, split=split)
        print(f"Rows: {len(ds)}")
        if len(ds) > 0:
            print("First row:", ds[0])
            print("Columns:", ds.column_names)
        else:
            print("[EMPTY SPLIT]")
    except Exception as e:
        print(f"[ERROR] Could not load split {split}: {e}")
