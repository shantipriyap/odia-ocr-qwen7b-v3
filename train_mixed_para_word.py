#!/usr/bin/env python3
"""
Phase 3: Mixed training with paragraph + word data.

WHY paragraph OCR FAILS today
──────────────────────────────
The current LoRA (checkpoint-4800) was trained exclusively on
`shantipriya/odia-ocr-merged` which contains word-level crops
(avg ~1-3 words per image).  The model learned:
    image_of_word → short_odia_string

When given a full-page paragraph image the model:
  1. Sees >500 visual tokens just for the image.
  2. Generates 2-5 Odia words and stops — matches training distribution.
  3. Even per-strip tiling fails because each strip (5 lines of text) is
     still out-of-distribution for a word-crop OCR model.

FIX
────
Mix paragraph samples from `OdiaGenAIOCR/Odia-lipi-ocr-data` into training
at a 20 % paragraph / 80 % word-level ratio.  This teaches the model to:
  • Recognise multi-line input as a cue to generate longer output.
  • Handle structural diversity (lines, paragraphs, columns).

This script runs on the H100 at 86.38.238.102 under /root/ml_env.
"""

import os, random, sys, json, textwrap
from dataclasses import dataclass, field
from typing import Optional


# === DATASET CONFIG ===
WORD_DS   = "shantipriya/odia-ocr-merged"       # 145 K word-level samples
SYN_DS    = "OdiaGenAIOCR/synthetic_data"       # synthetic word/line/para samples
KAGGLE_DS = "saswatsamal/odia-characters-dataset" # Kaggle dataset (word/char)
PARA_DS   = "OdiaGenAIOCR/Odia-lipi-ocr-data"   # 64 paragraph samples

import pathlib
LOCAL_OUT_DIR = "./checkpoints_para"
LOCAL_LOG_PATH = "./training_para.log"
OUT_DIR   = os.environ.get("ODIA_OCR_OUT_DIR", LOCAL_OUT_DIR)
LOG_PATH  = os.environ.get("ODIA_OCR_LOG_PATH", LOCAL_LOG_PATH)

# Ensure output/log directories exist
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
HF_TOKEN  = os.environ.get("HF_TOKEN", "os.getenv("HF_TOKEN", "")")

BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
PREV_CKPT   = "/root/odia_ocr/checkpoints_improved/checkpoint-4800"  # warm-start

PARA_RATIO  = 0.20    # fraction of each batch that are paragraph samples
BATCH_SIZE  = 2
GRAD_ACCUM  = 4       # effective batch = 8
MAX_STEPS   = 3000
WARMUP      = 100
LR          = 5e-5
MAX_WORD_TOKENS = 256
MAX_PARA_TOKENS = 1024   # longer output for paragraphs

# Paragraph images are large — resize to this max width before training
PARA_MAX_WIDTH = 960

# ────────────────────────────────────────────────────────────────────────────
import torch
from PIL import Image
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Odia OCR system. Extract all text from the given image "
    "exactly as it appears, preserving line breaks."
)

WORD_PROMPT  = "Extract the Odia text from this image. Return only the text."
PARA_PROMPT  = (
    "Extract all the Odia text from this image exactly as it appears, "
    "preserving line breaks. Return only the text, no explanation."
)




# ─── Data loading ────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor
def load_all_splits(ds_name):
    splits = []
    try:
        info = load_dataset(ds_name, streaming=False).keys()
        for split in ["train", "validation", "val", "test"]:
            if split in info:
                splits.append(split)
    except Exception:
        splits = ["train"]
    datasets = []
    for split in splits:
        try:
            datasets.append(load_dataset(ds_name, split=split, streaming=False, num_proc=4))
        except Exception:
            continue
    if datasets:
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets)
    return load_dataset(ds_name, split="train", streaming=False, num_proc=4)

log.info("Loading datasets in parallel …")
with ThreadPoolExecutor(max_workers=4) as executor:
    future_word = executor.submit(load_all_splits, WORD_DS)
    future_syn = executor.submit(load_all_splits, SYN_DS)
    future_para = executor.submit(load_all_splits, PARA_DS)
    future_kaggle = executor.submit(load_all_splits, KAGGLE_DS)
    word_ds = future_word.result()
    syn_ds = future_syn.result()
    para_ds = future_para.result()
    try:
        kaggle_ds = future_kaggle.result()
    except Exception as e:
        log.warning(f"Kaggle dataset not found or failed to load: {e}")
        kaggle_ds = None
log.info(f"  Word-level: {len(word_ds):,}  |  Synthetic: {len(syn_ds):,}  |  Paragraph: {len(para_ds):,}")

# Tag samples with their source and prompt
def tag_word(ex):
    return {**ex, "_source": "word", "_prompt": WORD_PROMPT, "_max_tokens": MAX_WORD_TOKENS}

def tag_syn(ex):
    return {**ex, "_source": "synthetic", "_prompt": WORD_PROMPT, "_max_tokens": MAX_WORD_TOKENS}

def tag_kaggle(ex):
    return {**ex, "_source": "kaggle", "_prompt": WORD_PROMPT, "_max_tokens": MAX_WORD_TOKENS}

def tag_para(ex):
    img = ex.get("image")
    if img is None:
        return None
    # Resize large paragraph images so visual tokens are manageable
    if hasattr(img, "size"):
        w, h = img.size
        if w > PARA_MAX_WIDTH:
            img = img.resize((PARA_MAX_WIDTH, int(h * PARA_MAX_WIDTH / w)), Image.LANCZOS)
    return {**ex, "image": img, "_source": "para", "_prompt": PARA_PROMPT, "_max_tokens": MAX_PARA_TOKENS}

word_ds = word_ds.map(tag_word, batched=False)
syn_ds = syn_ds.map(tag_syn, batched=False)
if kaggle_ds:
    kaggle_ds = kaggle_ds.map(tag_kaggle, batched=False)

para_rows = []
for i in range(len(para_ds)):
    row = tag_para(para_ds[i])
    if row:
        para_rows.append(row)
log.info(f"  Para samples after filtering: {len(para_rows)}")

# Augment paragraph data by repeating it to reach PARA_RATIO of total word+syn+kaggle size
total_word_like = len(word_ds) + len(syn_ds) + (len(kaggle_ds) if kaggle_ds else 0)
target_para = int(total_word_like * PARA_RATIO / (1 - PARA_RATIO))
para_repeated = []
while len(para_repeated) < target_para:
    para_repeated.extend(para_rows)
para_repeated = para_repeated[:target_para]
log.info(f"  Repeating {len(para_rows)} para samples → {len(para_repeated)} to balance")

# Build combined dataset
datasets_to_concat = [word_ds, syn_ds]
if kaggle_ds:
    datasets_to_concat.append(kaggle_ds)
para_dataset = Dataset.from_list(para_repeated)
datasets_to_concat.append(para_dataset)
combined = concatenate_datasets(datasets_to_concat).shuffle(seed=42)
log.info(f"  Combined dataset: {len(combined):,} samples ({PARA_RATIO*100:.0f}% para target)")


# ─── Model ───────────────────────────────────────────────────────────────────

# === Accelerate setup ===
accelerator = Accelerator()

log.info(f"Loading base model: {BASE_MODEL}")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Warm-start from previous checkpoint if available
if os.path.exists(PREV_CKPT):
    log.info(f"Warm-starting from {PREV_CKPT}")
    model = PeftModel.from_pretrained(base, PREV_CKPT, is_trainable=True)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
else:
    log.info("No previous checkpoint found — initialising fresh LoRA")
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

model.print_trainable_parameters()
model.enable_input_require_grads()

# Prepare model, processor, and data for distributed training
model, processor = accelerator.prepare(model, processor)


# ─── Data collator ───────────────────────────────────────────────────────────
class MixedOCRCollator:
    def __init__(self, processor):
        self.proc = processor

    def __call__(self, batch):
        images, texts, max_toks_list = [], [], []
        for ex in batch:
            try:
                img  = ex.get("image")
                text = ex.get("text", "").strip()
                src  = ex.get("_source", "word")
                prompt = ex.get("_prompt", WORD_PROMPT)
                max_t  = ex.get("_max_tokens", MAX_WORD_TOKENS)

                if img is None or not text:
                    continue
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    continue

                # Build conversation for chat template
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text",  "text": prompt},
                        ],
                    },
                    {"role": "assistant", "content": text},
                ]

                text_input = self.proc.apply_chat_template(
                    msgs[:-1], tokenize=False, add_generation_prompt=True
                )
                text_input += text + self.proc.tokenizer.eos_token

                images.append(img)
                texts.append(text_input)
                max_toks_list.append(max_t)
            except Exception:
                continue

        if not images:
            return {"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[-100]])}

        try:
            # Use batch processing for efficiency
            max_tok = max(max_toks_list)
            inputs = self.proc(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            # Mask out padding tokens in labels
            labels = inputs["input_ids"].clone()
            labels[labels == self.proc.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels
            return inputs
        except Exception as e:
            log.warning(f"Collator error: {e}")
            return {"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[-100]])}


# ─── Training ────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    optim="adamw_torch",
    bf16=True,
    report_to=[],
    eval_strategy="no",
    seed=42,
    push_to_hub=False,
)

log.info("Creating Trainer …")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined,
    data_collator=MixedOCRCollator(processor),
)

log.info("=" * 70)
log.info("STARTING MIXED PARAGRAPH+WORD TRAINING")
log.info(f"  Steps: {MAX_STEPS}  |  Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
log.info(f"  Word: {len(word_ds):,}  |  Para (augmented): {len(para_repeated):,}")
log.info(f"  Output: {OUT_DIR}")
log.info("=" * 70)

trainer.train()

# Save final checkpoint and push
final_path = os.path.join(OUT_DIR, "final")
trainer.save_model(final_path)
log.info(f"Saved to {final_path}")

# Push to HF Hub
try:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    repo_id = "shantipriya/odia-ocr-qwen-finetuned_v3_para"
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=final_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Phase 3: Mixed word+paragraph OCR training",
    )
    log.info(f"Pushed to https://huggingface.co/{repo_id}")
except Exception as e:
    log.warning(f"Push failed: {e}")

log.info("DONE")
