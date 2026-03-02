import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# CONFIG
BASE_MODEL = "shantipriya/hunyuan-ocr-odia"
PEFT_MODEL = "shantipriya/hunyuan-ocr-odia"
DATASET = "shantipriya/odia-ocr-merged"

# Load processor and model
processor = AutoProcessor.from_pretrained(BASE_MODEL)
base_model = AutoModelForVision2Seq.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL)
model.eval()

# Load a small eval set
eval_ds = load_dataset(DATASET, split="test")

# Evaluate on first 10 samples
for i, example in enumerate(eval_ds.select(range(10))):
    image = example["image"]
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all Odia text from this image. Return only the Odia text."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Sample {i+1} Prediction: {pred}")
    print(f"Ground Truth: {example.get('extracted_text') or example.get('text')}")
    print("-"*40)
