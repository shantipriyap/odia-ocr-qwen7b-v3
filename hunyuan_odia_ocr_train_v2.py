"""
HunyuanOCR Fine-tuning + Evaluation for Odia OCR
Base model : tencent/HunyuanOCR
Datasets   : OdiaGenAIOCR/synthetic_data   (columns: image, extracted_text)
             OdiaGenAIOCR/odia-ocr-merged   (columns: image, text)
Output     : shantipriya/hunyuan-ocr-odia  (pushed to HF Hub)

Requirements (run on server before starting):
  pip install peft datasets accelerate evaluate jiwer pillow tqdm huggingface_hub
  pip install "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"

Run:
  cd /root/hunyuan_odia && python hunyuan_odia_ocr_train_v2.py
"""

import os
import sys
import json
import contextlib
import torch
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_MODEL    = "tencent/HunyuanOCR"
HF_REPO_ID    = "shantipriya/hunyuan-ocr-odia"
HF_TOKEN      = "os.getenv("HF_TOKEN", "")"

DATASET_SYNTHETIC = "OdiaGenAIOCR/synthetic_data"   # image + extracted_text  (5,349 rows)
DATASET_MERGED    = "OdiaGenAIOCR/odia-ocr-merged"  # image + text            (73,400 rows)

OCR_PROMPT    = "Extract all Odia text from this image. Return only the Odia text, nothing else."

OUTPUT_DIR    = "/root/hunyuan_odia/output"
MAX_STEPS     = 1000
EVAL_STEPS    = 100
SAVE_STEPS    = 200
LEARNING_RATE = 1e-4
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
WARMUP_STEPS  = 50
MAX_NEW_TOKENS = 512
EVAL_SAMPLES  = 200

LORA_R        = 32
LORA_ALPHA    = 64
LORA_DROPOUT  = 0.05
# Include MLP layers (gate/up/down_proj) for sufficient capacity to learn Odia script
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# ENSURE HUNYUAN TRANSFORMERS SUPPORT
# ──────────────────────────────────────────────
try:
    from transformers import HunYuanVLForConditionalGeneration
    print("[OK] HunYuanVLForConditionalGeneration available.")
except ImportError:
    print("[SETUP] Installing patched transformers for HunyuanOCR support...")
    os.system(
        'pip install -q "git+https://github.com/huggingface/transformers'
        '@82a06db03535c49aa987719ed0746a76093b1ec4"'
    )
    from transformers import HunYuanVLForConditionalGeneration

from transformers import AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model

# ──────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────
MAX_IMG_SIZE = 512  # Limit image size to avoid HunyuanVL processor token index overflow

def ensure_pil(img, max_size=MAX_IMG_SIZE):
    pil = None
    if img is None:
        return None
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, bytes):
        from io import BytesIO
        pil = Image.open(BytesIO(img)).convert("RGB")
    elif isinstance(img, str) and os.path.exists(img):
        pil = Image.open(img).convert("RGB")
    else:
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                pil = Image.fromarray(img).convert("RGB")
        except Exception:
            pass
    if pil is None:
        return None
    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    try:
        import jiwer
        cer = jiwer.cer(references, predictions)
        wer = jiwer.wer(references, predictions)
    except ImportError:
        print("[WARN] jiwer not installed. Skipping CER/WER.")
        cer, wer = None, None

    exact = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    result = {
        "exact_match": round(exact / len(predictions), 4) if predictions else 0,
        "num_samples": len(predictions),
    }
    if cer is not None:
        result["CER"] = round(cer, 4)
        result["WER"] = round(wer, 4)
    return result


# ──────────────────────────────────────────────
# LOAD & MERGE DATASETS
# ──────────────────────────────────────────────
def load_and_merge_datasets():
    print("\n[DATA] Loading OdiaGenAIOCR/synthetic_data ...")
    ds1 = load_dataset(DATASET_SYNTHETIC, split="train")
    print(f"  -> {len(ds1)} rows | cols: {ds1.column_names}")

    print("[DATA] Loading OdiaGenAIOCR/odia-ocr-merged ...")
    ds2 = load_dataset(DATASET_MERGED, split="train")
    print(f"  -> {len(ds2)} rows | cols: {ds2.column_names}")

    # Normalize both datasets to: image + gt_text
    ds1 = ds1.map(
        lambda x: {"image": x["image"], "gt_text": x["extracted_text"]},
        remove_columns=ds1.column_names,
        desc="Normalize synthetic_data",
    )
    ds2 = ds2.map(
        lambda x: {"image": x["image"], "gt_text": x["text"]},
        remove_columns=ds2.column_names,
        desc="Normalize odia-ocr-merged",
    )

    combined = concatenate_datasets([ds1, ds2]).shuffle(seed=42)
    # Filter out rows with missing text to prevent empty collator batches
    combined = combined.filter(
        lambda x: x["gt_text"] is not None and len((x["gt_text"] or "").strip()) > 0,
        desc="Filter empty rows",
    )
    total = len(combined)
    val_size  = min(500, int(total * 0.02))
    test_size = min(500, int(total * 0.02))

    train_ds = combined.select(range(0, total - val_size - test_size))
    val_ds   = combined.select(range(total - val_size - test_size, total - test_size))
    test_ds  = combined.select(range(total - test_size, total))

    print(f"\n[DATA] Splits -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# ──────────────────────────────────────────────
# DATA COLLATOR
# ──────────────────────────────────────────────
@dataclass
class OdiaOCRCollator:
    processor: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict]) -> Dict:
        valid_images, texts_input = [], []

        for sample in batch:
            image = ensure_pil(sample.get("image"))
            gt    = (sample.get("gt_text") or "").strip()
            if image is None or not gt:
                continue

            # FIX: Include GT text as assistant response so the model trains on it
            messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  OCR_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": gt,
                },
            ]
            try:
                # add_generation_prompt=False: we include the full GT response
                full_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                valid_images.append(image)
                texts_input.append(full_text)
            except Exception as e:
                print(f"[WARN] Skipping sample (apply_chat_template): {e}")
                continue

        if not valid_images:
            return {}

        # Retry with progressively smaller images if processor raises IndexError
        for attempt in range(3):
            try:
                inputs = self.processor(
                    text=texts_input,
                    images=valid_images,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                break
            except (IndexError, Exception) as e:
                print(f"[WARN] Processor attempt {attempt+1} failed: {e}. Shrinking images.")
                valid_images = [img.resize((img.width // 2, img.height // 2), Image.LANCZOS) for img in valid_images]
        else:
            return {}

        # Build labels — mask prompt tokens, train only on GT text + hy_Assistant token.
        # HunyuanOCR chat template structure:
        #   [BOS][image_tokens][OCR_PROMPT]<｜hy_User｜>[GT_TEXT]<｜hy_Assistant｜>
        # Everything up to and including <｜hy_User｜> (ID 120006) is prompt → masked.
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id

        HY_USER_ID = 120006  # <｜hy_User｜> ends the user/prompt turn
        for i in range(labels.shape[0]):
            ids = labels[i].tolist()
            # Find the LAST <｜hy_User｜> token — mask everything up to and including it
            found = False
            for j in range(len(ids) - 1, -1, -1):
                if ids[j] == HY_USER_ID:
                    labels[i, : j + 1] = -100
                    found = True
                    break
            if not found:
                # Fallback: mask all (sample had no user turn boundary)
                print(f"[WARN] sample {i}: <|hy_User|> not found, masking all tokens")
                labels[i, :] = -100

        # Also mask padding tokens
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Debug: print label stats on first call to verify masking is correct
        if not getattr(self, '_debug_printed', False):
            self._debug_printed = True
            for i in range(min(2, labels.shape[0])):
                total = labels.shape[1]
                masked = (labels[i] == -100).sum().item()
                active = total - masked
                print(f"[DEBUG] sample {i}: {total} total tokens, {masked} masked, {active} active (GT) tokens")
        inputs["labels"] = labels
        return dict(inputs)


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def run_evaluation(model, processor, dataset, num_samples=EVAL_SAMPLES, tag="eval"):
    model.eval()
    predictions, references = [], []

    for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"[EVAL] {tag}"):
        sample = dataset[idx]
        image  = ensure_pil(sample.get("image"))
        gt     = (sample.get("gt_text") or "").strip()
        if image is None or not gt:
            continue

        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  OCR_PROMPT},
                ],
            },
        ]
        texts  = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        inputs = processor(text=texts, images=[image], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        input_ids = inputs.get("input_ids", gen_ids)
        trimmed   = [out[len(inp):] for inp, out in zip(input_ids, gen_ids)]
        pred      = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        predictions.append(pred)
        references.append(gt)

    metrics = compute_metrics(predictions, references)
    print(f"\n[EVAL] {tag}: {metrics}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"eval_{tag}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "samples": [
                {"pred": p, "ref": r}
                for p, r in zip(predictions[:20], references[:20])
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_file}")
    return metrics


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"HunyuanOCR Odia Fine-tuning")
    print(f"Base  : {BASE_MODEL}")
    print(f"Output: {HF_REPO_ID}")
    print(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    print(f"{'='*60}\n")

    # 1. Load datasets
    train_ds, val_ds, test_ds = load_and_merge_datasets()

    # 2. Load processor
    print(f"\n[MODEL] Loading processor from {BASE_MODEL}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=False)

    # 3. Load model
    print(f"[MODEL] Loading HunYuanVLForConditionalGeneration...")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        attn_implementation="eager",
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.config.use_cache = False
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # 4. Apply LoRA
    print("\n[LORA] Applying LoRA...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5. Baseline evaluation
    print("\n[EVAL] Baseline (before training)...")
    run_evaluation(model, processor, val_ds, num_samples=50, tag="baseline")

    # 6. Training
    collator = OdiaOCRCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    print(f"\n[TRAIN] Starting ({MAX_STEPS} steps)...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    print("[TRAIN] Complete.")

    # 7. Save locally
    print(f"\n[SAVE] Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Fix adapter_config.json base_model_name_or_path
    cfg_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["base_model_name_or_path"] = BASE_MODEL
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  adapter_config.json base_model set to '{BASE_MODEL}'")

    # 8. Final test evaluation
    print("\n[EVAL] Final evaluation on test set...")
    test_metrics = run_evaluation(model, processor, test_ds, num_samples=EVAL_SAMPLES, tag="test")

    # 9. Push to Hub
    print(f"\n[PUSH] Pushing to {HF_REPO_ID}...")
    model.push_to_hub(HF_REPO_ID, token=HF_TOKEN, commit_message="Fine-tuned HunyuanOCR on Odia OCR (synthetic + merged)")
    processor.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    results_path = os.path.join(OUTPUT_DIR, "eval_test.json")
    api.upload_file(
        path_or_fileobj=results_path,
        path_in_repo="eval_results.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add evaluation results",
    )

    print(f"\n{'='*60}")
    print(f"DONE! https://huggingface.co/{HF_REPO_ID}")
    print(f"Test Metrics: {test_metrics}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
