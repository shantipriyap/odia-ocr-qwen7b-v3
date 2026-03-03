"""
Create a composite OCR sample image for the blog post.
Shows: Input Image | Ground Truth | Prediction | Status
"""

from PIL import Image, ImageDraw, ImageFont
import os, json, textwrap

# ── load eval data (use first 10 samples) ───────────────────────────────────
EVAL_JSON = "/Users/shantipriya/work/odia_ocr/inference_samples/eval_ck4400.json"
with open(EVAL_JSON) as f:
    data = json.load(f)

# pick 5 exact-match (good) + 2 near (mixed) + 3 misses (bad) from the 50 samples
samples_raw = data["samples"]
good    = [s for s in samples_raw if s["exact"]][:5]
mixed   = [s for s in samples_raw if (not s["exact"] and s["cer"] <= 0.15)][:2]
bad     = [s for s in samples_raw if (not s["exact"] and s["cer"] > 0.15)][:3]
selected = good + mixed + bad           # 10 rows

# ── sample images ordered 1-10 ──────────────────────────────────────────────
IMG_DIR = "/Users/shantipriya/work/odia_ocr/inference_samples"
img_files = [os.path.join(IMG_DIR, f"sample_{i:02d}.png") for i in range(1, 11)]

# ── layout constants ─────────────────────────────────────────────────────────
ROW_H      = 88
THUMB_W    = 360     # max width for the thumbnail column
COL_GT_W   = 280
COL_PRED_W = 280
COL_STS_W  = 60
PADDING    = 12
TOTAL_W    = THUMB_W + COL_GT_W + COL_PRED_W + COL_STS_W + PADDING * 5
HEADER_H   = 52

NUM_ROWS   = len(selected)
TOTAL_H    = HEADER_H + ROW_H * NUM_ROWS + PADDING * (NUM_ROWS + 1)

# ── colours ──────────────────────────────────────────────────────────────────
BG          = (248, 249, 250)
HEADER_BG   = (36, 41, 47)
HEADER_FG   = (255, 255, 255)
ROW_ALT     = (255, 255, 255)
ROW_DARK    = (241, 243, 246)
BORDER      = (210, 215, 220)
GREEN       = (40, 167, 69)
ORANGE      = (255, 140, 0)
RED         = (220, 53, 69)
TEXT_DARK   = (33, 37, 41)
TEXT_LABEL  = (108, 117, 125)

# ── fonts ─────────────────────────────────────────────────────────────────────
ODIA_FONT_REGULAR = "/tmp/NotoSansOriya-Regular.ttf"
ODIA_FONT_BOLD    = "/tmp/NotoSansOriya-Bold.ttf"

def load_latin_font(size, bold=False):
    """Load a Latin/UI font (for labels, headers)."""
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

def load_odia_font(size, bold=False):
    """Load Noto Sans Oriya for rendering Odia Unicode text."""
    path = ODIA_FONT_BOLD if (bold and os.path.exists(ODIA_FONT_BOLD)) else ODIA_FONT_REGULAR
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return load_latin_font(size, bold)  # fallback

font_header  = load_latin_font(18, bold=True)
font_label   = load_latin_font(13, bold=True)
font_text    = load_latin_font(14)
font_odia    = load_odia_font(18)      # Noto Sans Oriya for Odia Unicode text
font_small   = load_latin_font(12)

# ── canvas ────────────────────────────────────────────────────────────────────
canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), BG)
draw   = ImageDraw.Draw(canvas)

# ── draw header bar ───────────────────────────────────────────────────────────
draw.rectangle([0, 0, TOTAL_W, HEADER_H], fill=HEADER_BG)
title = "Odia OCR  ·  Qwen2.5-VL-3B Fine-tuned  ·  Model Inference Samples"
tx, ty = TOTAL_W // 2, HEADER_H // 2
bbox = draw.textbbox((0, 0), title, font=font_header)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
draw.text((tx - tw // 2, ty - th // 2), title, fill=HEADER_FG, font=font_header)

# ── column header row ─────────────────────────────────────────────────────────
col_x = [
    PADDING,                                                   # image col
    PADDING * 2 + THUMB_W,                                     # GT col
    PADDING * 3 + THUMB_W + COL_GT_W,                         # pred col
    PADDING * 4 + THUMB_W + COL_GT_W + COL_PRED_W,            # status col
]
COL_HDR_H = 28
col_hdr_y  = HEADER_H
draw.rectangle([0, col_hdr_y, TOTAL_W, col_hdr_y + COL_HDR_H], fill=(60, 65, 70))
for lbl, cx, cw in [
    ("Input Image",     col_x[0], THUMB_W),
    ("Ground Truth",    col_x[1], COL_GT_W),
    ("Prediction",      col_x[2], COL_PRED_W),
    ("",                col_x[3], COL_STS_W),
]:
    b = draw.textbbox((0, 0), lbl, font=font_label)
    tw2 = b[2] - b[0]
    draw.text((cx + (cw - tw2) // 2, col_hdr_y + 7), lbl, fill=(220, 225, 230), font=font_label)

HEADER_TOT = HEADER_H + COL_HDR_H

# ── draw rows ─────────────────────────────────────────────────────────────────
for row_idx, (s, img_path) in enumerate(zip(selected, img_files)):
    row_y = HEADER_TOT + PADDING + row_idx * (ROW_H + PADDING)
    row_bg = ROW_ALT if row_idx % 2 == 0 else ROW_DARK
    draw.rectangle([0, row_y, TOTAL_W, row_y + ROW_H], fill=row_bg)

    # thin border bottom
    draw.line([0, row_y + ROW_H, TOTAL_W, row_y + ROW_H], fill=BORDER, width=1)

    # ── thumbnail ────────────────────────────────────────────────────────────
    try:
        img = Image.open(img_path).convert("RGB")
        ratio = min(THUMB_W / img.width, (ROW_H - 8) / img.height)
        nw, nh = int(img.width * ratio), int(img.height * ratio)
        thumb = img.resize((nw, nh), Image.LANCZOS)
        tx_p = col_x[0] + (THUMB_W - nw) // 2
        ty_p = row_y + (ROW_H - nh) // 2
        canvas.paste(thumb, (tx_p, ty_p))
        # subtle border around thumbnail
        draw.rectangle([tx_p - 1, ty_p - 1, tx_p + nw, ty_p + nh], outline=BORDER)
    except Exception as e:
        draw.text((col_x[0] + 4, row_y + 30), f"[img error: {e}]", fill=RED, font=font_small)

    # ── GT text ──────────────────────────────────────────────────────────────
    gt_text  = s["gt"]
    pred_text = s["pred"]
    exact    = s["exact"]
    cer      = s["cer"]

    def draw_text_centered_v(text, cx, cy, cw, color, font):
        b = draw.textbbox((0, 0), text, font=font)
        tw3 = b[2] - b[0]
        th3 = b[3] - b[1]
        draw.text((cx + max(0, (cw - tw3) // 2), cy - th3 // 2), text, fill=color, font=font)

    mid_y = row_y + ROW_H // 2

    draw_text_centered_v(gt_text,   col_x[1], mid_y, COL_GT_W,   TEXT_DARK, font_odia)
    draw_text_centered_v(pred_text, col_x[2], mid_y, COL_PRED_W, TEXT_DARK, font_odia)

    # ── status indicator ─────────────────────────────────────────────────────
    if exact:
        status_color = GREEN
        status_sym   = "✓"
        status_lbl   = "exact"
    elif cer <= 0.15:
        status_color = ORANGE
        status_sym   = "~"
        status_lbl   = f"CER {cer:.0%}"
    else:
        status_color = RED
        status_sym   = "✗"
        status_lbl   = f"CER {cer:.0%}"

    # coloured pill background
    pill_x = col_x[3] + 4
    pill_y = row_y + ROW_H // 2 - 20
    draw.rounded_rectangle([pill_x, pill_y, pill_x + COL_STS_W - 8, pill_y + 40],
                           radius=8, fill=(*status_color, 30))
    draw_text_centered_v(status_sym, col_x[3], mid_y - 7, COL_STS_W, status_color, font_header)
    draw_text_centered_v(status_lbl, col_x[3], mid_y + 14, COL_STS_W, status_color, font_small)

# ── footer ────────────────────────────────────────────────────────────────────
foot_y = TOTAL_H - 22
draw.rectangle([0, foot_y, TOTAL_W, TOTAL_H], fill=(36, 41, 47))
foot_txt = "Model: shantipriya/odia-ocr-qwen-finetuned_v2  ·  Dataset: shantipriya/odia-ocr-merged (145K samples)"
b = draw.textbbox((0, 0), foot_txt, font=font_small)
fw = b[2] - b[0]
draw.text(((TOTAL_W - fw) // 2, foot_y + 5), foot_txt, fill=(170, 175, 180), font=font_small)

# ── save ─────────────────────────────────────────────────────────────────────
OUT = "/Users/shantipriya/work/odia_ocr/blog/ocr_sample_results.png"
canvas.save(OUT, "PNG", optimize=True)
print(f"Saved  →  {OUT}")
print(f"Canvas size: {canvas.size}")

# also save a 2x smaller version for web embedding
small = canvas.resize((TOTAL_W // 2, TOTAL_H // 2), Image.LANCZOS)
OUT2 = "/Users/shantipriya/work/odia_ocr/blog/ocr_sample_results_small.png"
small.save(OUT2, "PNG", optimize=True)
print(f"Saved  →  {OUT2}")
