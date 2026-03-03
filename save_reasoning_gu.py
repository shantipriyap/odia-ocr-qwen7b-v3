from datasets import Dataset

# Read input and output files
input_file = "input.txt"
output_file = "output.txt"

records = []
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "r", encoding="utf-8") as fout:
    for inp, out in zip(fin, fout):
        records.append({
            "input": inp.strip(),
            "output": out.strip()
        })

# Create Hugging Face Dataset and save to disk
hf_ds = Dataset.from_list(records)
hf_ds.save_to_disk("Reasoning_GU")
print("Saved Hugging Face dataset to Reasoning_GU/")
