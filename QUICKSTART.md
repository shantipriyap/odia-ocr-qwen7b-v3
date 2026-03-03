# Quick Start Guide

## Installation

### Requirements
- Python 3.8+
- GPU with at least 12GB VRAM (recommended: A100 with 80GB)
- ~50GB disk space for model and data

### Setup

**Option 1: Automated Setup (Recommended)**

```bash
bash setup.sh
```

**Option 2: Manual Setup**

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Training

Train the model on Odia OCR dataset (58,720 samples, 3 epochs).

### Before Training
- Ensure you have **GPU with 80GB+ VRAM** (A100 recommended)
- Training takes ~4 hours on A100
- Dataset: `shantipriya/odia-ocr-merged` (auto-downloaded)

### Run Training

```bash
python train.py
```

### Output
- Checkpoints saved to: `./checkpoint-odia-qwen/`
- Saves every 500 steps
- Training log: `training_YYYYMMDD_HHMMSS.log`

### Training Configuration
```python
# Defaults in train.py:
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
MAX_STEPS = 3500  # ~3 epochs
LEARNING_RATE = 2e-4
```

### Monitor Training

```bash
# Watch training progress
tail -f training_*.log

# Check GPU usage
nvidia-smi -l 1  # Refresh every 1 second
```

---

## Evaluation

Evaluate model performance on test set.

### Run Evaluation

```bash
# Evaluate full dataset
python eval.py

# Evaluate first 100 samples only
python eval.py --max-samples 100
```

### Output
- Metrics: CER (Character Error Rate), WER (Word Error Rate), Exact Match
- Results saved to: `eval_results_YYYYMMDD_HHMMSS.json`
- Evaluation log: `evaluation_YYYYMMDD_HHMMSS.log`

### Sample Output

```
📊 EVALUATION RESULTS
=================================================================
✅ Processed: 100 samples
❌ Failed: 0 samples

📈 Metrics:
   Average CER: 25.34%
   Average WER: 18.92%
   Exact Match Rate: 45.00%
```

---

## Inference

Extract Odia text from images using the trained model.

### Single Image

```bash
python inference.py --image document.jpg
```

Output:
```
✅ Model loaded successfully
Extracted Odia:
ଓଡ଼ିଆ ସାହିତ୍ୟର ଇତିହାସ ଅତ୍ୟନ୍ତ ସମୃଦ୍ଧ
```

### With Raw Output

```bash
python inference.py --image document.jpg --raw
```

Shows both raw model output (with chat template) and extracted Odia text.

### Batch Processing (Directory)

```bash
python inference.py --directory ./images
```

### Save Results to File

```bash
python inference.py --directory ./images --output results.json
```

Results format:
```json
{
  "image": "images/doc1.jpg",
  "result": "ଓଡ଼ିଆ ସାହିତ୍ୟ...",
  "success": true
}
```

### Using Custom Model

```bash
python inference.py --image document.jpg --model your-model-id
```

---

## Common Issues & Troubleshooting

### 🔴 Out of Memory (OOM)
**Problem:** `CUDA out of memory` error during training

**Solutions:**
```bash
# Reduce batch size in train.py
BATCH_SIZE = 1  # Already minimum
GRADIENT_ACCUMULATION_STEPS = 1  # Reduce accumulation

# Or use CPU (slow):
# Modify train.py: device_map="cpu"
```

### 🔴 Model Download Fails
**Problem:** `Connection error downloading model`

**Solution:**
```bash
# Set HuggingFace cache
export HF_HOME=/path/to/cache
python train.py
```

### 🔴 Dataset Not Found
**Problem:** `ConnectionError: Couldn't reach https://huggingface.co`

**Solution:**
```bash
# Download dataset manually first
python -c "from datasets import load_dataset; load_dataset('shantipriya/odia-ocr-merged')"
```

### 🔴 No GPU Found
**Problem:** `RuntimeError: CUDA is not available`

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Tips

### Training Speed
```python
# Already optimized in train.py:
- Gradient checkpointing (30-40% memory savings)
- bfloat16 precision (faster)
- Batch size 1 with accumulation (memory efficient)
```

### Inference Speed
```bash
# For faster inference, use smaller batch size or quantization
# See inference.py for options
```

---

## Docker Usage

### Build Image
```bash
docker build -t odia-ocr .
```

### Run Container
```bash
docker run --gpus all -v $(pwd):/workspace odia-ocr python train.py
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `train.py` | Fine-tune model on 58.7K Odia samples |
| `eval.py` | Evaluate CER, WER, exact match metrics |
| `inference.py` | OCR inference (single/batch) |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Automated setup script |
| `Dockerfile` | Container configuration |

---

## Resources

- **Dataset:** [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged)
- **Model:** [shantipriya/odia-ocr-qwen-finetuned](https://huggingface.co/shantipriya/odia-ocr-qwen-finetuned)
- **Base Model:** [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

---

## Next Steps

1. ✅ Install dependencies: `bash setup.sh`
2. ✅ Run inference: `python inference.py --image test.jpg`
3. 📊 Evaluate model: `python eval.py`
4. 🔄 Fine-tune: `python train.py` (requires A100 GPU)

**Questions?** Check the main [README.md](README.md) for more details.
