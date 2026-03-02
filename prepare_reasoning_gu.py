import json
from datasets import Dataset

input_file = "input.txt"
output_file = "output.txt"
hf_dir = "Reasoning_GU"
jsonl_file = "reasoning_gu.jsonl"

# Read input and output lines
with open(input_file, "r", encoding="utf-8") as fin:
    inputs = [line.strip() for line in fin if line.strip()]
with open(output_file, "r", encoding="utf-8") as fout:
    outputs = [line.strip() for line in fout if line.strip()]

assert len(inputs) == len(outputs), "Input and output line counts do not match!"

# Write JSONL file
with open(jsonl_file, "w", encoding="utf-8") as fj:
    for inp, out in zip(inputs, outputs):
        fj.write(json.dumps({"input": inp, "output": out}, ensure_ascii=False) + "\n")
print(f"Wrote {jsonl_file} with {len(inputs)} records.")

# Create and save Hugging Face dataset
records = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]
ds = Dataset.from_list(records)
ds.save_to_disk(hf_dir)
print(f"Saved Hugging Face dataset to {hf_dir}/")
