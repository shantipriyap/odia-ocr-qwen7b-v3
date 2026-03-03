import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import torch
from tqdm import tqdm

# Parameters
model_name = "ai4bharat/IndicTrans3-beta"
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_repo = "OdiaGenAIdata/Reasoning_GU"
hf_token = "os.getenv("HF_TOKEN", "")"
batch_size = 16  # Increase if you have more GPU memory

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

print("Translating and cleaning data in true batches...")
rows = [row for row in ds_ref if isinstance(row, dict)]
total = len(rows)

for start in tqdm(range(0, total, batch_size), desc="Batch Processing", unit="batch"):
    end = min(start + batch_size, total)
    batch_rows = rows[start:end]
    batch_records = [row.copy() for row in batch_rows]
    for col in columns_to_translate:
        # Gather all texts for this column in the batch
        texts = [r.get(col, "") if isinstance(r.get(col, ""), str) else "" for r in batch_records]
        # Identify which need translation (non-empty)
        to_translate = [i for i, t in enumerate(texts) if t.strip()]
        if to_translate:
            input_texts = [texts[i] for i in to_translate]
            # Tokenize and translate as a batch
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200)
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Assign translations back
            for idx, rec_idx in enumerate(to_translate):
                batch_records[rec_idx][f"{col}_gu"] = translations[idx]
        # For empty/invalid, set blank
        for i in range(len(batch_records)):
            if not texts[i].strip():
                batch_records[i][f"{col}_gu"] = ""
    clean_records.extend(batch_records)

print(f"Total clean records: {len(clean_records)}")

# Save as a single split and push to HF
hf_ds = Dataset.from_list(clean_records)
print("Pushing cleaned dataset to Hugging Face (split='train')...")
hf_ds.push_to_hub(hf_repo, token=hf_token, private=False, split="train")
print(f"Done! Cleaned data available at https://huggingface.co/datasets/{hf_repo}")
