#!/usr/bin/env python3
"""
Evaluate checkpoint-500 on Odia OCR test dataset
"""

import os
import sys
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
from jiwer import cer, wer
from tqdm import tqdm
import time

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
CHECKPOINT_DIR = "/root/odia_ocr/qwen_ocr_finetuned"
CHECKPOINT_NAME = "checkpoint-500"
TEST_SIZE = 50  # Evaluate on 50 samples for consistency

print("\n" + "="*80)
print("📊 ODIA OCR CHECKPOINT-500 EVALUATION")
print("="*80)

# Load dataset
print(f"\n[1/4] 📥 Loading test dataset ({TEST_SIZE} samples)...")
dataset = load_dataset(DATASET_NAME, split="train")

# Filter dataset to only include samples with text longer than 5 characters
dataset = dataset.filter(lambda x: x.get("text") and len(str(x.get("text"))) > 5)
print(f"   Filtered dataset to {len(dataset)} samples with text > 5 chars")

np.random.seed(42)  # For reproducibility
indices = np.random.choice(len(dataset), min(TEST_SIZE, len(dataset)), replace=False)
test_dataset = dataset.select(indices.tolist())
print(f"✅ Loaded {len(test_dataset)} test samples")

# Load model
print(f"\n[2/4] 📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)
print(f"✅ Base model loaded")

# Load LoRA adapter
checkpoint_path = f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}"
if os.path.exists(checkpoint_path):
    print(f"   Loading LoRA adapter: {checkpoint_path}...")
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        torch_dtype=torch.float16
    )
    print(f"✅ LoRA adapter loaded from {CHECKPOINT_NAME}")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

model.eval()

# Helper function (needed for processor)
def process_vision_info(messages):
    image_inputs = []
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    # Get the image value directly
                    img = item.get("image")
                    if img is not None:
                        image_inputs.append(img)
    return image_inputs if image_inputs else None, None

# Prepare prompt
PROMPT = """Extract all text from this image. Return only the text content, nothing else."""

# Run evaluation
print(f"\n[3/4] 🔍 Running inference on {TEST_SIZE} samples...")
predictions = []
references = []
inference_times = []

for idx, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
    try:
        image = sample["image"]
        reference_text = sample.get("text") or sample.get("character")
        
        # Debug first few samples
        if idx < 3:
            print(f"\nSample {idx}: text type={type(reference_text)}, value={reference_text[:50] if reference_text else 'None'}")
        
        if reference_text is None or not str(reference_text).strip():
            print(f"⚠️  Sample {idx}: Skipping - no text")
            continue
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        inference_time = time.time() - start_time
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        predicted_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        predictions.append(predicted_text)
        references.append(reference_text)
        inference_times.append(inference_time)
        
    except Exception as e:
        print(f"\n⚠️  Error on sample {idx}: {e}")
        # Don't append anything for failed samples
        continue

# Calculate metrics
print(f"\n[4/4] 📈 Calculating metrics...")
try:
    # Filter out None values from predictions and references
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                   if ref is not None and pred is not None and ref.strip() and pred.strip()]
    
    if not valid_pairs:
        print("❌ No valid prediction-reference pairs found")
        sys.exit(1)
    
    valid_references, valid_predictions = zip(*valid_pairs)
    print(f"   Valid samples: {len(valid_pairs)}/{len(references)}")
    
    # CER
    cer_score = cer(list(valid_references), list(valid_predictions)) * 100
    
    # WER
    wer_score = wer(list(valid_references), list(valid_predictions)) * 100
    
    # Character Accuracy
    char_accuracy = 100 - cer_score
    
    # Exact matches
    exact_matches = sum(1 for ref, pred in valid_pairs 
                       if ref.strip().lower() == pred.strip().lower())
    exact_match_rate = (exact_matches / len(valid_pairs)) * 100
    
    # Inference stats
    avg_inference_time = np.mean(inference_times)
    throughput = 1.0 / avg_inference_time
    
    # Results
    results = {
        "checkpoint": CHECKPOINT_NAME,
        "test_samples": len(valid_pairs),
        "metrics": {
            "character_error_rate": round(cer_score, 2),
            "character_accuracy": round(char_accuracy, 2),
            "word_error_rate": round(wer_score, 2),
            "exact_match_rate": round(exact_match_rate, 2),
            "avg_inference_time": round(avg_inference_time, 3),
            "throughput": round(throughput, 3)
        },
        "sample_predictions": [
            {"reference": ref[:100], "prediction": pred[:100]}
            for ref, pred in list(valid_pairs)[:5]
        ]
    }
    
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS - CHECKPOINT-500")
    print("="*80)
    print(f"\n{'Metric':<30} {'Value':<15} {'Status'}")
    print("-" * 80)
    print(f"{'Character Error Rate (CER)':<30} {cer_score:.2f}%")
    print(f"{'Character Accuracy':<30} {char_accuracy:.2f}%")
    print(f"{'Word Error Rate (WER)':<30} {wer_score:.2f}%")
    print(f"{'Exact Match Rate':<30} {exact_match_rate:.2f}%")
    print(f"{'Avg Inference Time':<30} {avg_inference_time:.3f} sec")
    print(f"{'Throughput':<30} {throughput:.3f} samples/sec")
    print("="*80)
    
    # Save results
    output_file = f"/root/odia_ocr/evaluation_checkpoint500.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
except Exception as e:
    print(f"❌ Error calculating metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
