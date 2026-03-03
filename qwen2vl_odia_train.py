from datasets import load_dataset

from transformers import AutoProcessor, Trainer
from PIL import Image
# Use a valid public Qwen2.5-VL checkpoint
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

# --- Load your dataset ---

# Load both datasets and concatenate
from datasets import concatenate_datasets


# Load only the first 10 samples from each dataset for fast test
train1 = load_dataset("shantipriya/odia-ocr-merged", split="train").select(range(10))
train2 = load_dataset("OdiaGenAIOCR/synthetic_data", split="train").select(range(10))

# Filter invalid samples in each small subset
def filter_invalid(example):
    return example.get("text") not in [None, ""] and example.get("image") is not None

train1 = train1.filter(filter_invalid)
train2 = train2.filter(filter_invalid)

train_dataset = concatenate_datasets([train1, train2])

# For evaluation, use the same small set (or a subset)
eval_dataset = train_dataset.select(range(min(10, len(train_dataset))))

# --- Filter out invalid samples (empty text or missing images) ---
def filter_invalid(example):
    return example["text"] not in [None, ""] and example["image"] is not None

train_dataset = train_dataset.filter(filter_invalid)
eval_dataset  = eval_dataset.filter(filter_invalid)

print(f"Train samples after filtering: {len(train_dataset)}")
print(f"Eval samples after filtering: {len(eval_dataset)}")

# Limit to first 10 samples for quick test run
train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))

# Limit to first 10 samples for quick test run
train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))



import os
# Add missing constants for training
SAVE_STEPS = 500
HF_REPO_ID = "shantipriya/odia-ocr-qwen-finetuned"
# --- Test: Visual feature extraction sanity check ---
def test_image_features(model, batch):
    pixel_values = batch["pixel_values"]  # shape: (B, 3, H, W)
    outputs = model.get_image_features(pixel_values, return_dict=True)
    print("Image features shape:", outputs.pooler_output.shape)

# Example usage (uncomment for manual debugging):
# for batch in train_dataloader:
#     test_image_features(model, batch)
# Batch sanity check utility
def check_batch(batch):
    """
    Sanity check for a batch before feeding to Qwen2VL model.
    batch: dict containing pixel_values, input_ids, attention_mask, labels
    """
    pixel_values = batch.get("pixel_values")
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    import torch

from datasets import load_dataset
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

# Add missing import for TrainingArguments
from transformers import TrainingArguments

# Set base model and dtype if not already set
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
import torch
DTYPE = torch.bfloat16

# Load processor and model
processor = AutoProcessor.from_pretrained(BASE_MODEL)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True, torch_dtype=DTYPE, device_map="auto")


 # Full fine-tuning (no LoRA/PEFT, pure model update)
print("Trainable parameters:")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable} || all params: {total} || trainable%: {trainable/total:.4%}")

# Data collator
class OCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        images, texts = [], []
        for example in batch:
            image = example["image"]
            # Always use 'text' key for Qwen2VL prompt, fallback to empty string if missing
            text = example.get("text")
            if text is None:
                text = ""
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            images.append(image)
            texts.append(text)

        assert len(images) == len(texts), f"Batch size mismatch: {len(images)} images, {len(texts)} texts"

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

# --- QwenVLCollator (ONLY ONCE, TOP-LEVEL) ---
class QwenVLCollator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        # Remove samples with missing image
        batch = [x for x in batch if x.get("image") is not None]
        if len(batch) == 0:
            raise ValueError("Empty batch after filtering!")

        images = [x["image"] for x in batch]
        # Always construct a valid prompt with <|image|> placeholder
        texts = [
            (x.get("text") if x.get("text") not in [None, ""] else "Extract all Odia text from this image: <|image|>")
            if "<|image|>" in (x.get("text") or "")
            else "Extract all Odia text from this image: <|image|>"
            for x in batch
        ]

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


# --- Data collator (NO INDENT!) ---
data_collator = OCRDataCollator(processor)

# --- TrainingArguments (minimal, adjust as needed) ---
training_args = TrainingArguments(
    output_dir="./qwen2vl-odia",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    remove_unused_columns=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Train & Validate
train_result = trainer.train()
metrics = train_result.metrics
print("\n=== TRAINING METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Final eval
eval_metrics = trainer.evaluate()
print("\n=== FINAL EVAL METRICS ===")
for k, v in eval_metrics.items():
    print(f"{k}: {v}")

trainer.push_to_hub("Final Qwen2.5-VL Odia model (full fine-tune)")
