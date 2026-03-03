import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# CONFIG
# ==============================
SRC_DATASET = "Yourgotoguy/Gujarati_Reasoning_4800"
SRC_SPLIT = "train"

HF_REPO = "OdiaGenAIdata/Reasoning_GU"
HF_TOKEN = os.environ.get("HF_TOKEN")  # ✅ SAFE

MODEL_NAME = "ai4bharat/IndicTrans3-beta"

BATCH_SIZE = 16
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 256

OUTPUT_JSONL = "reasoning_gu.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN environment variable not set")

# ==============================
# LOAD MODEL
# ==============================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

model.eval()

# ==============================
# TRANSLATION
# ==============================
def translate_en_to_gu(texts):
    out = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        prompts = [
            f"<en> <2gu> {t}"
            for t in batch
            if isinstance(t, str) and t.strip()
        ]

        if not prompts:
            out.extend([""] * len(batch))
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                do_sample=False
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        out.extend(decoded)

    return out


# ==============================
# MAIN
# ==============================
def main():
    ds = load_dataset(SRC_DATASET, split=SRC_SPLIT)

    columns = ["user_input", "reasoning", "target_language"]

    rows = [r for r in ds if isinstance(r, dict)]

    translated_records = [{} for _ in range(len(rows))]

    for col in columns:
        texts = [r.get(col, "") for r in rows]
        translated = translate_en_to_gu(texts)

        for i, t in enumerate(translated):
            translated_records[i][f"{col}_gu"] = t

    # Save JSONL (optional but useful)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in translated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hf_ds = Dataset.from_list(translated_records)

    hf_ds.push_to_hub(
        HF_REPO,
        token=HF_TOKEN,
        split="train",
        private=False
    )


if __name__ == "__main__":
    main()