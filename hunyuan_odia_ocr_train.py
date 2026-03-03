import sys
if sys.version_info < (3, 12):
    sys.exit("ERROR: Python 3.12 or higher is required for HunYuanVLForConditionalGeneration. Please install Python 3.12+ and rerun this script.")
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from transformers import HunYuanVLForConditionalGeneration

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# =========================
# CONFIG
# =========================
BASE_MODEL = "Qwen/Qwen2.5-VL"
HF_REPO_ID = "shantipriya/hunyuan-ocr-odia"

DATASET_1 = "shantipriya/odia-ocr-merged"
DATASET_2 = "OdiaGenAIOCR/synthetic_data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

SAVE_STEPS = 500
MAX_NEW_TOKENS = 512

# =========================
# LOAD DATASETS (NO MAP!)
# =========================
print("Loading datasets…")

ds1 = load_dataset(DATASET_1, split="train")
ds2 = load_dataset(DATASET_2, split="train")

train_dataset = concatenate_datasets([ds1, ds2])
# Limit to first 10 samples for debugging
train_dataset = train_dataset.select(range(10))

# Prepare a small eval set (5 samples from DATASET_1 test split)
eval_dataset = load_dataset(DATASET_1, split="test").select(range(5))

print("Total training samples:", len(train_dataset))

# =========================
# LOAD MODEL
# =========================
processor = AutoProcessor.from_pretrained(BASE_MODEL)

model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    device_map="auto"
)

# =========================
# LORA
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# DATA COLLATOR (CRITICAL)
# =========================
class OCRDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = []
        prompts = []
        labels = []

        for example in batch:
            image = example["image"]
            text = example.get("extracted_text") or example.get("text")

            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": "Extract all Odia text from this image. Return only the Odia text."
                        },
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            images.append(image)
            prompts.append(prompt)
            labels.append(text)

        model_inputs = self.processor(
            text=prompts,
            images=images,
            padding="max_length",
            max_length=MAX_NEW_TOKENS,
            return_tensors="pt"
        )

        with self.processor.tokenizer.as_target_tokenizer():
            label_ids = self.processor.tokenizer(
                labels,
                padding="max_length",
                max_length=MAX_NEW_TOKENS,
                return_tensors="pt"
            ).input_ids

        model_inputs["labels"] = label_ids
        return model_inputs

data_collator = OCRDataCollator(processor)

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    push_to_hub=True,
    hub_model_id=HF_REPO_ID,
    hub_strategy="checkpoint",
    remove_unused_columns=False,
    report_to="none",
    evaluation_strategy="epoch",  # Evaluate after each epoch
    eval_steps=None,
)

# =========================
# TRAINER
# =========================

# Use Hugging Face token from environment if available
import os
from huggingface_hub import login
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# =========================
# TRAIN & VALIDATE
# =========================
train_result = trainer.train()

# Log train and eval metrics
metrics = train_result.metrics
print("\n=== TRAINING METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

eval_metrics = trainer.evaluate()
print("\n=== FINAL EVAL METRICS ===")
for k, v in eval_metrics.items():
    print(f"{k}: {v}")

trainer.push_to_hub("Final Odia OCR LoRA model")
# FINAL PUSH
