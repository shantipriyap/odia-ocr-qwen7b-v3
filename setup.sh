#!/usr/bin/env bash
# Quick Start Script for Odia OCR

set -e

echo "🚀 Odia OCR - Quick Setup Guide"
echo "================================="
echo ""

# Check Python
echo "1️⃣  Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
echo "✅ Python OK"
echo ""

# Create virtual environment
echo "2️⃣  Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "3️⃣  Activating virtual environment..."
source venv/bin/activate
echo "✅ Activated: $(which python3)"
echo ""

# Install requirements
echo "4️⃣  Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
echo ""

# Show available commands
echo "📋 Available Commands:"
echo ""
echo "   Training:"
echo "   python train.py"
echo ""
echo "   Evaluation:"
echo "   python eval.py"
echo ""
echo "   Inference (single image):"
echo "   python inference.py --image document.jpg"
echo ""
echo "   Inference (directory):"
echo "   python inference.py --directory ./images"
echo ""
echo "✨ Setup complete! You're ready to go."
