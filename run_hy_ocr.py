from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import torch
from io import BytesIO

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text

def get_image(input_source):
    if isinstance(input_source, Image.Image):
        return input_source
    elif isinstance(input_source, bytes):
        return Image.open(BytesIO(input_source))
    elif isinstance(input_source, str) and input_source.startswith(('http://', 'https://')):
        import requests
        response = requests.get(input_source)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(input_source)

def main():
    # Use your Odia dataset and prompt
    DATASET = "shantipriya/odia-ocr-merged"
    model_name_or_path = "shantipriya/hunyuan-ocr-odia"
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    # Load one sample from test split
    eval_ds = load_dataset(DATASET, split="test")
    example = eval_ds[0]
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
    print(f"Prediction: {pred}")
    print(f"Ground Truth: {example.get('extracted_text') or example.get('text')}")
    print("-"*40)

if __name__ == '__main__':
    main()
