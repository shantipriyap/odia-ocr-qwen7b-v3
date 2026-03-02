import os
from datasets import load_from_disk

input_file = "input.txt"
output_file = "output.txt"
hf_dir = "Reasoning_GU"

# Compare line counts
def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

input_lines = count_lines(input_file)
output_lines = count_lines(output_file)

print(f"Input lines: {input_lines}")
print(f"Output lines: {output_lines}")

if input_lines == output_lines:
    print("Line counts match.")
else:
    print("Line counts DO NOT match!")

# Compare file sizes
input_size = os.path.getsize(input_file)
output_size = os.path.getsize(output_file)
print(f"Input file size: {input_size} bytes")
print(f"Output file size: {output_size} bytes")

# Check Hugging Face dataset size
if os.path.exists(hf_dir):
    ds = load_from_disk(hf_dir)
    print(f"Hugging Face dataset records: {len(ds)}")
else:
    print(f"Hugging Face dataset directory '{hf_dir}' not found.")
