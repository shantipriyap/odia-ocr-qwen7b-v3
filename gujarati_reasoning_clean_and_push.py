import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import torch
from tqdm import tqdm

# Parameters
model_name = "ai4bharat/IndicTrans3-beta"
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_repo = "OdiaGenAIdata/Reasoning_GU"
hf_token = "os.getenv("HF_TOKEN", "")"

# Load source dataset
print("Loading source dataset...")
ds_ref = load_dataset("Yourgotoguy/Gujarati_Reasoning_4800", split="train")
print(f"Loaded {len(ds_ref)} rows.")

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Gemma3ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

columns_to_translate = ["user_input", "reasoning", "target_language"]
clean_records = []

print("Translating and cleaning data...")
for row in tqdm(ds_ref, desc="Processing", unit="row"):
    # Only process dict-like rows
    if not isinstance(row, dict):
        continue
    record = row.copy()
    for col in columns_to_translate:
        orig = record.get(col, "")
        if orig and isinstance(orig, str) and orig.strip():
            inputs = tokenizer(orig, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            translated = ""
        record[f"{col}_gu"] = translated
    clean_records.append(record)

print(f"Total clean records: {len(clean_records)}")

# Save as a single split and push to HF
hf_ds = Dataset.from_list(clean_records)
print("Pushing cleaned dataset to Hugging Face (split='train')...")
hf_ds.push_to_hub(hf_repo, token=hf_token, private=False, split="train")
print(f"Done! Cleaned data available at https://huggingface.co/datasets/{hf_repo}")
