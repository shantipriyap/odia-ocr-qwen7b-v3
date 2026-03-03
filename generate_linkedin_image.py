from PIL import Image, ImageDraw, ImageFont

# Use a sample image from your workspace
sample_image_path = "inference_samples/sample_01.png"
output_text = "ଏହା ହେଉଛି ଓଡ଼ିଆ ଓସିଆର ମଡେଲର ଉତ୍ପାଦନ"  # Replace with your model's output

# Load sample image and resize for layout
sample_img = Image.open(sample_image_path).convert("RGB").resize((300, 300))

# Create blank image for output text
output_img = Image.new("RGB", (300, 300), color="white")
draw = ImageDraw.Draw(output_img)
try:
    font = ImageFont.truetype("NotoSansOriya-Regular.ttf", 32)  # Use a font that supports Odia
except:
    font = ImageFont.load_default()
draw.text((20, 120), output_text, fill="black", font=font)

# Combine images side by side
combined = Image.new("RGB", (620, 340), color="white")
combined.paste(sample_img, (10, 30))
combined.paste(output_img, (310, 30))

# Add title and footer
draw_combined = ImageDraw.Draw(combined)
title = "Odia OCR Breakthrough!"
subtitle = "Fine-tuned Qwen2.5-VL for Odia Text Recognition"
footer = "Model: shantipriya/odia-ocr-qwen-finetuned_v2 (Hugging Face)"

draw_combined.text((10, 0), title, fill="navy", font=font)
draw_combined.text((10, 20), subtitle, fill="gray", font=font)
draw_combined.text((10, 320), footer, fill="darkgreen", font=font)

combined.save("odia_ocr_linkedin_post.png")
print("Image saved as odia_ocr_linkedin_post.png")
