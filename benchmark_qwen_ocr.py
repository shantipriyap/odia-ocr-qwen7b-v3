import sys
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "Iftesha/odia-ocr-benchmark"

print("="*70)
print(f"🔍 BENCHMARKING MODEL: {MODEL_ID}")
print(f"🗂️  DATASET: {DATASET_ID}")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"


print("📥 Loading dataset...")
# Load benchmark dataset (use 'train' split if 'test' is not available)
dataset = load_dataset(DATASET_ID)
split_name = 'test' if 'test' in dataset else 'train'
test_dataset = dataset[split_name]
print(f"   ✅ {len(test_dataset):,} samples loaded from split: '{split_name}'")
print(f"✅ {len(dataset):,} samples loaded")

print("📦 Loading model and processor...")
from transformers import AutoTokenizer, AutoImageProcessor
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    from transformers import ProcessorMixin
    class CustomProcessor(ProcessorMixin):
        attributes = ["tokenizer", "image_processor"]
        tokenizer_class = type(tokenizer)
        image_processor_class = type(image_processor)
        def __init__(self, tokenizer, image_processor):
            self.tokenizer = tokenizer
            self.image_processor = image_processor
        def __call__(self, images, text, **kwargs):
            inputs = self.image_processor(images, return_tensors="pt")
            encodings = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            inputs.update(encodings)
            return inputs
    processor = CustomProcessor(tokenizer, image_processor)
except Exception as e:
    print(f"❌ Processor workaround failed: {e}")
    raise
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
model.eval()

correct = 0
results = []

for ex in tqdm(test_dataset, desc="Evaluating"):
    image = ex.get("image")
    gt = ex.get("text") or ex.get("extracted_text")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        continue
    inputs = processor(images=image, text="", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
    pred = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    results.append({"gt": gt, "pred": pred})
    if pred == gt:
        correct += 1

accuracy = correct / len(results) if results else 0
print(f"\n{'='*70}")
print(f"✅ BENCHMARK COMPLETE")
print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{len(results)})")
print(f"{'='*70}")

# Optionally, save detailed results
def save_results():
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Results saved to benchmark_results.json")

if __name__ == "__main__":
    if "--save" in sys.argv:
        save_results()
