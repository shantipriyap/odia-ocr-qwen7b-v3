#!/usr/bin/env python3
"""Gradio Space for Odia OCR using Qwen2.5-VL + adapter."""
import os
import uuid
import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

BASE_MODEL = os.environ.get("HF_OCR_BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
ADAPTER_ID = os.environ.get("HF_OCR_ADAPTER", "OdiaGenAIOCR/odia-ocr-qwen-finetuned")
MAX_TOKENS = int(os.environ.get("HF_OCR_MAX_TOKENS", "512"))
PROMPT = "Extract all Odia text from this image. Return only the text."

print(f"Loading base model: {BASE_MODEL}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="cuda",
    trust_remote_code=True,
)

if ADAPTER_ID:
    print(f"Loading adapter: {ADAPTER_ID}")
    model.load_adapter(ADAPTER_ID)

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)


@spaces.GPU
def perform_ocr(image):
    if image is None:
        return "No image provided."

    # Gradio gives numpy; convert to PIL
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Qwen VL expects messages structure
    src = f"/tmp/{uuid.uuid4().hex}.png"
    image.save(src)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": src},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, use_cache=True)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    try:
        os.remove(src)
    except OSError:
        pass

    return output_text.strip()


with gr.Blocks(title="Odia OCR (Qwen2.5-VL)") as app:
    gr.Markdown("# Odia OCR (Qwen2.5-VL + Adapter)")
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload Image", type="numpy")
            btn = gr.Button("Extract Text", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Extracted Text", lines=18, show_copy_button=True)

    btn.click(perform_ocr, inputs=image, outputs=output)
    image.change(perform_ocr, inputs=image, outputs=output)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
