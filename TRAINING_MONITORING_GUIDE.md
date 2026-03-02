# 🚀 Odia OCR Retraining - Monitoring Guide

## Current Status (Live)

**Training started:** ~12:42 UTC  
**Model:** Qwen2.5-VL-3B-Instruct + LoRA (rank=64)  
**Dataset:** 145K+ Odia OCR samples (merged)  
**GPU:** NVIDIA A100 SXM4 (80GB)  

---

## 📊 Live Metrics Tracking

### Phase 1: Data Preprocessing (Current - In Progress)
Currently at **~34% complete** in Map stage (20,209/58,720 examples)

**To monitor:**
```bash
# SSH to remote server
ssh root@95.216.229.232

# Real-time tail of training log
tail -f /root/odia_ocr/training_fixed.log

# Or use the monitor script (updates every 60 sec)
cat /root/odia_ocr/monitor.log
```

### Phase 2: Training Loop (Expected ~13:20 UTC)
Once data preprocessing completes, you'll see:
```
[1/500] loss = X.XXXX
[2/500] loss = X.XXXX
...
```

**Target metrics to watch:**
- Loss should start high (5-8) and decrease
- By step 50, expect loss ~2-4
- By step 500, expect loss converging to ~0.5-1.5

---

## 🔍 Key Metrics Explained

### **Loss (Training Loss)**
- Measures how well model fits the training data
- Decreases over time = good
- Chart: Should show smooth downward trend

### **Character Error Rate (CER)**
- Percentage of characters incorrectly predicted
- CER = (Substitutions + Deletions + Insertions) / Reference_Length
- Lower is better (0% = perfect)
- Typical OCR: 5-15% CER is good

### **Word Error Rate (WER)**
- Similar to CER but at word level
- Useful for whole-word accuracy

### **Exact Match Accuracy**
- Percentage of samples where full text matches perfectly
- Most stringent metric
- Typical: 40-70% for OCR

---

## 📈 Real-Time Monitoring Commands

### Check Current Progress
```bash
ssh root@95.216.229.232 'tail -20 /root/odia_ocr/training_fixed.log | grep -E "Map|Filter|Step|loss"'
```

### Get Status Every 30 Seconds
```bash
ssh root@95.216.229.232 'watch -n 30 "tail -5 /root/odia_ocr/training_fixed.log"'
```

### Extract All Loss Values
```bash
ssh root@95.216.229.232 'grep "loss = " /root/odia_ocr/training_fixed.log'
```

### Check GPU Usage
```bash
ssh root@95.216.229.232 'nvidia-smi'
```

### Process Resource Usage
```bash
ssh root@95.216.229.232 'ps aux | grep train_fixed | head -6'
```

---

## 📊 Expected Timeline

| Time | Phase | Status |
|------|-------|--------|
| 12:42 | Start | Training script launched, model loading |
| 12:50 | LoRA Load | PEFT adapters materialized to GPU |
| 13:10 | Map → 70% | Data mapping progressing |
| 13:20 | Map → 100% | Data preprocessing complete |
| 13:22 | Filter | Removing invalid samples |
| 13:25 | **Training Begins** | First loss metrics visible ✨ |
| 13:30 | Step ~10 | Training in full swing |
| 14:00 | Step ~60 | Loss stabilizing |
| ~14:45 | Step ~300 | Mid-training checkpoint |
| ~15:30 | **Step 500** | **Training Complete** 🎉 |

---

## 🎯 Next Steps After Training Completes

### 1. Save Model Checkpoint
Checkpoint saved automatically at: `/root/outputs/checkpoint-500`

### 2. Evaluate on Test Set
```bash
# Local: Run evaluation script on your MacBook
# (requires GPU or will run slowly on CPU)
python evaluate_training_results.py
```

This will compute:
- **CER** (Character Error Rate)
- **WER** (Word Error Rate)  
- **Accuracy** (Exact match %)

### 3. Upload to HuggingFace (Optional)
```bash
python /root/final_push.py
```

---

## 💡 Troubleshooting

### If training seems stuck:
```bash
# Check if processes are running
ps aux | grep train_fixed

# Check for errors in log
tail -100 /root/odia_ocr/training_fixed.log | tail -20
```

### If GPU memory errors appear:
- Reduce batch size or gradient_accumulation_steps
- Check if other processes are using GPU memory

### If data loading is very slow:
- This is normal for 145K images
- Each worker loads, decodes, and processes images
- Speed varies based on I/O and CPU load

---

## 📝 Log Interpretation

### Normal Output Patterns

**Loading phase:**
```
Loading weights: 39%|███▉      | 320/824
Loading weights: 100%|██████████| 824/824
```

**Data mapping:**
```
Map (num_proc=4):  34%|███▍      | 20209/58720 [12:34<41:34, 15.44 examples/s]
```

**Training:**
```
[10/500] loss = 4.2356 (0.8567 sec/step)
[11/500] loss = 4.1234 (0.8234 sec/step)
```

---

## 🔔 Notifications Setup

To get email/slack alerts when training completes:
- Check monitor.log periodically
- Or write a simple wrapper script that sends alerts

---

**Last Updated:** During active training  
**Questions?** Check training_fixed.log for detailed diagnostics
