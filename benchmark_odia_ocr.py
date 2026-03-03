import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
from PIL import Image

MODEL_NAME = "shantipriya/odia-ocr-qwen-finetuned_v2"
BENCHMARK_DATASET = "Iftesha/odia-ocr-benchmark"

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on device: {device}")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True
)
model.eval()

# Load benchmark dataset
ds = load_dataset(BENCHMARK_DATASET, split="test")
print(f"Loaded benchmark dataset with {len(ds)} samples.")

correct = 0
results = []

for ex in tqdm(ds, desc="Benchmarking"):
    image = ex["image"]
    gt_text = ex["text"]
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        continue
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    pred_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    results.append({"gt": gt_text, "pred": pred_text})
    if pred_text == gt_text:
        correct += 1

accuracy = correct / len(ds)
print(f"\nBenchmark accuracy: {accuracy:.4f} ({correct}/{len(ds)})")

# Save detailed results
import csv
with open("benchmark_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["gt", "pred"])
    writer.writeheader()
    writer.writerows(results)
print("Detailed results saved to benchmark_results.csv")
