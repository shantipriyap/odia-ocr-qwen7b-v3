#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — one-time server setup for Odia OCR Phase 3 training
# Run: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "============================================================"
echo "  Odia OCR Phase 3 — Server Setup  (2× A100-80GB)"
echo "============================================================"

# 1. System packages
apt-get update -qq && apt-get install -y -qq git wget curl tmux htop nvtop

# 2. Python venv
python3 -m venv /root/venv
source /root/venv/bin/activate

# 3. PyTorch (CUDA 12.x for A100)
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# 4. Training stack
pip install -q \
    transformers>=4.50 \
    peft>=0.14 \
    accelerate>=0.34 \
    datasets>=2.21 \
    huggingface_hub>=0.24 \
    packaging \
    sentencepiece \
    editdistance \
    Pillow \
    tqdm

# 5. Flash-Attention 2 (pre-built wheel for A100 + CUDA 12.1 + Python 3.10)
pip install flash-attn --no-build-isolation -q || \
    echo "[WARN] flash-attn build failed — will use eager attention (slower but works)"

# 6. Create working directory
mkdir -p /root/phase3_paragraph/output_2gpu

echo ""
echo "[OK] Setup complete."
echo "     Activate: source /root/venv/bin/activate"
echo "     Then run: bash launch.sh"
