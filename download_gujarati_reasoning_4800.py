from datasets import load_dataset, Dataset
import os
import json

# Download Gujarati_Reasoning_4800 and extract 'input' column
ds = load_dataset("Yourgotoguy/Gujarati_Reasoning_4800", split="train")

# Save all 'input' lines to input.txt
with open("input.txt", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(row["input"].strip() + "\n")
print(f"Wrote input.txt with {len(ds)} lines.")

# (Optional) Save all 'output' lines to output.txt if you want to compare
with open("output.txt", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(row["output"].strip() + "\n")
print(f"Wrote output.txt with {len(ds)} lines.")

# Save as Hugging Face dataset (for upload)
ds.save_to_disk("Reasoning_GU_reference")
print("Saved reference dataset to Reasoning_GU_reference/")
