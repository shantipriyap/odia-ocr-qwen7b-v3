"""
Fix: Ensure each sample image is paired with its correct ground truth and prediction by parsing results.html.
"""
import os
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

HTML = "/Users/shantipriya/work/odia_ocr/inference_samples/results.html"
IMG_DIR = "/Users/shantipriya/work/odia_ocr/inference_samples"
ODIA_FONT = "/tmp/NotoSansOriya-Regular.ttf"
ODIA_FONT_BOLD = "/tmp/NotoSansOriya-Bold.ttf"

# Parse HTML for image, GT, pred
with open(HTML, encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

cards = soup.find_all("div", class_="card")
samples = []
for card in cards:
    img = card.find("img")
    info = card.find("div", class_="info")
    if not img or not info:
        continue
    texts = info.find_all("div", class_="text")
    if len(texts) < 2:
        continue
    gt = texts[0].text.strip()
    pred = texts[1].text.strip()
    badge = info.find(class_="badge").text.strip() if info.find(class_="badge") else ""
    cer = info.find(class_="cer").text.strip() if info.find(class_="cer") else ""
    samples.append({
        "img": img["src"],
        "gt": gt,
        "pred": pred,
        "badge": badge,
        "cer": cer
    })
# Sort samples by image filename (e.g., sample_01.png)
samples.sort(key=lambda s: s["img"]) 

# Layout
ROW_H, THUMB_W, COL_GT_W, COL_PRED_W, COL_STS_W, PADDING = 88, 360, 280, 280, 60, 12
TOTAL_W = THUMB_W + COL_GT_W + COL_PRED_W + COL_STS_W + PADDING * 5
HEADER_H = 52
NUM_ROWS = len(samples)
TOTAL_H = HEADER_H + ROW_H * NUM_ROWS + PADDING * (NUM_ROWS + 1)
BG = (248, 249, 250)
HEADER_BG = (36, 41, 47)
HEADER_FG = (255, 255, 255)
ROW_ALT = (255, 255, 255)
ROW_DARK = (241, 243, 246)
BORDER = (210, 215, 220)
GREEN = (40, 167, 69)
ORANGE = (255, 140, 0)
RED = (220, 53, 69)
TEXT_DARK = (33, 37, 41)
TEXT_LABEL = (108, 117, 125)

def load_odia_font(size, bold=False):
    path = ODIA_FONT_BOLD if (bold and os.path.exists(ODIA_FONT_BOLD)) else ODIA_FONT
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def load_latin_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

font_header = load_latin_font(18, bold=True)
font_label = load_latin_font(13, bold=True)
font_odia = load_odia_font(18)
font_small = load_latin_font(12)

canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), BG)
draw = ImageDraw.Draw(canvas)

draw.rectangle([0, 0, TOTAL_W, HEADER_H], fill=HEADER_BG)
title = "Odia OCR  ·  Qwen2.5-VL-3B Fine-tuned  ·  Model Inference Samples"
bbox = draw.textbbox((0, 0), title, font=font_header)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
draw.text(((TOTAL_W - tw) // 2, HEADER_H // 2 - th // 2), title, fill=HEADER_FG, font=font_header)

col_x = [PADDING, PADDING * 2 + THUMB_W, PADDING * 3 + THUMB_W + COL_GT_W, PADDING * 4 + THUMB_W + COL_GT_W + COL_PRED_W]
COL_HDR_H = 28
col_hdr_y = HEADER_H
draw.rectangle([0, col_hdr_y, TOTAL_W, col_hdr_y + COL_HDR_H], fill=(60, 65, 70))
for lbl, cx, cw in [
    ("Input Image", col_x[0], THUMB_W),
    ("Ground Truth", col_x[1], COL_GT_W),
    ("Prediction", col_x[2], COL_PRED_W),
    ("", col_x[3], COL_STS_W),
]:
    b = draw.textbbox((0, 0), lbl, font=font_label)
    tw2 = b[2] - b[0]
    draw.text((cx + (cw - tw2) // 2, col_hdr_y + 7), lbl, fill=(220, 225, 230), font=font_label)

HEADER_TOT = HEADER_H + COL_HDR_H
for row_idx, s in enumerate(samples):
    row_y = HEADER_TOT + PADDING + row_idx * (ROW_H + PADDING)
    row_bg = ROW_ALT if row_idx % 2 == 0 else ROW_DARK
    draw.rectangle([0, row_y, TOTAL_W, row_y + ROW_H], fill=row_bg)
    draw.line([0, row_y + ROW_H, TOTAL_W, row_y + ROW_H], fill=BORDER, width=1)
    img_path = os.path.join(IMG_DIR, s["img"])
    try:
        img = Image.open(img_path).convert("RGB")
        ratio = min(THUMB_W / img.width, (ROW_H - 8) / img.height)
        nw, nh = int(img.width * ratio), int(img.height * ratio)
        thumb = img.resize((nw, nh), Image.LANCZOS)
        tx_p = col_x[0] + (THUMB_W - nw) // 2
        ty_p = row_y + (ROW_H - nh) // 2
        canvas.paste(thumb, (tx_p, ty_p))
        draw.rectangle([tx_p - 1, ty_p - 1, tx_p + nw, ty_p + nh], outline=BORDER)
    except Exception as e:
        draw.text((col_x[0] + 4, row_y + 30), f"[img error: {e}]", fill=RED, font=font_small)
    def draw_text_centered_v(text, cx, cy, cw, color, font):
        b = draw.textbbox((0, 0), text, font=font)
        tw3 = b[2] - b[0]
        th3 = b[3] - b[1]
        draw.text((cx + max(0, (cw - tw3) // 2), cy - th3 // 2), text, fill=color, font=font)
    mid_y = row_y + ROW_H // 2
    draw_text_centered_v(s["gt"], col_x[1], mid_y, COL_GT_W, TEXT_DARK, font_odia)
    draw_text_centered_v(s["pred"], col_x[2], mid_y, COL_PRED_W, TEXT_DARK, font_odia)
    badge = s["badge"].lower()
    if "exact" in badge:
        status_color, status_sym, status_lbl = GREEN, "✓", "exact"
    elif "near" in badge or "partial" in badge:
        status_color, status_sym, status_lbl = ORANGE, "~", s["cer"]
    else:
        status_color, status_sym, status_lbl = RED, "✗", s["cer"]
    pill_x = col_x[3] + 4
    pill_y = row_y + ROW_H // 2 - 20
    draw.rounded_rectangle([pill_x, pill_y, pill_x + COL_STS_W - 8, pill_y + 40], radius=8, fill=(*status_color, 30))
    draw_text_centered_v(status_sym, col_x[3], mid_y - 7, COL_STS_W, status_color, font_header)
    draw_text_centered_v(status_lbl, col_x[3], mid_y + 14, COL_STS_W, status_color, font_small)
foot_y = TOTAL_H - 22
draw.rectangle([0, foot_y, TOTAL_W, TOTAL_H], fill=(36, 41, 47))
foot_txt = "Model: shantipriya/odia-ocr-qwen-finetuned_v2  ·  Dataset: shantipriya/odia-ocr-merged (145K samples)"
b = draw.textbbox((0, 0), foot_txt, font=font_small)
fw = b[2] - b[0]
draw.text(((TOTAL_W - fw) // 2, foot_y + 5), foot_txt, fill=(170, 175, 180), font=font_small)
OUT = "/Users/shantipriya/work/odia_ocr/blog/ocr_sample_results.png"
canvas.save(OUT, "PNG", optimize=True)
print(f"Saved  →  {OUT}")
small = canvas.resize((TOTAL_W // 2, TOTAL_H // 2), Image.LANCZOS)
OUT2 = "/Users/shantipriya/work/odia_ocr/blog/ocr_sample_results_small.png"
small.save(OUT2, "PNG", optimize=True)
print(f"Saved  →  {OUT2}")
