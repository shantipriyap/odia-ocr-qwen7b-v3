#!/usr/bin/env python3
"""
monitor.py — Live training monitor for Odia OCR Phase 3
=========================================================
Shows:
  - GPU utilisation + VRAM (both A100s)
  - Latest training loss from log
  - HuggingFace Hub latest commit
  - Quick 3-sample inference from the live adapter

Usage:
  python3 monitor.py [--interval 60] [--hf-repo shantipriya/odia-ocr-qwen7b-phase3]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH   = Path("/root/phase3_paragraph/train.log")
OUTPUT_DIR = Path("/root/phase3_paragraph/output_2gpu")
ADAPTER    = OUTPUT_DIR / "final_adapter"

SYNTHETIC_DATASET = "OdiaGenAIOCR/synthetic_data"
OCR_PROMPT = (
    "Extract all Odia text from this image exactly as written, "
    "preserving line order and paragraph structure. "
    "Return only the Odia text, nothing else."
)


# ─────────────────────────────────────────────────────────────────────────────
def gpu_stats() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        )
        lines = []
        for row in out.strip().split("\n"):
            idx, name, util, mem_used, mem_total, temp = [x.strip() for x in row.split(",")]
            bar = "█" * int(int(util) // 10) + "░" * (10 - int(util) // 10)
            lines.append(
                f"  GPU{idx} [{bar}] {util:>3}%  "
                f"VRAM {mem_used}/{mem_total} MiB  {temp}°C"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"  nvidia-smi error: {e}"


def latest_loss_from_log() -> str:
    if not LOG_PATH.exists():
        return "  log not found"
    try:
        text = LOG_PATH.read_text(errors="ignore")
        # Match trainer log lines: {'loss': 1.234, 'step': 100, ...}
        matches = re.findall(r"'loss':\s*([\d.]+).*?'step':\s*(\d+)", text)
        if not matches:
            # Try alternate HF Trainer format
            matches = re.findall(r"\{'loss':\s*([\d.]+)[^}]*'step':\s*(\d+)\}", text)
        if matches:
            loss, step = matches[-1]
            return f"  Step {step:>6}  Loss {loss}"
        return "  No loss entries yet"
    except Exception as e:
        return f"  Log read error: {e}"


def log_tail(n: int = 6) -> str:
    if not LOG_PATH.exists():
        return ""
    try:
        lines = LOG_PATH.read_text(errors="ignore").strip().split("\n")
        return "\n".join(f"  {l}" for l in lines[-n:])
    except Exception:
        return ""


def hf_latest_commit(repo_id: str) -> str:
    try:
        from huggingface_hub import HfApi
        api  = HfApi()
        info = api.list_repo_commits(repo_id, repo_type="model", limit=1)
        c    = list(info)[0]
        return f"  {c.created_at.strftime('%H:%M:%S')}  {c.title[:60]}"
    except Exception as e:
        return f"  HF commit check failed: {e}"


def quick_inference(repo_id: str) -> str:
    """Load latest pushed adapter and run on 3 random synthetic samples."""
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
        from PIL import Image

        BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
        proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, repo_id)
        model.eval()

        ds = load_dataset(SYNTHETIC_DATASET, split="train").shuffle(seed=int(time.time()) % 1000)
        results = []
        for sample in ds.select(range(3)):
            from io import BytesIO
            img = sample["image"]
            if isinstance(img, bytes):
                img = Image.open(BytesIO(img)).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = img.convert("RGB")

            gt = (sample.get("extracted_text") or "").strip()[:80]

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": OCR_PROMPT},
            ]}]
            tp = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[tp], images=[img], return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=256, do_sample=False)
            pred = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()[:80]
            results.append(f"  GT  : {gt}\n  PRED: {pred}")

        del model, base
        torch.cuda.empty_cache()
        return "\n".join(results)
    except Exception as e:
        return f"  Inference error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval",  type=int,  default=60,  help="Refresh interval seconds")
    parser.add_argument("--hf-repo",   type=str,  default="shantipriya/odia-ocr-qwen7b-phase3")
    parser.add_argument("--no-infer",  action="store_true",    help="Skip inference (faster)")
    args = parser.parse_args()

    print(f"[MONITOR] Refreshing every {args.interval}s  |  Ctrl+C to stop\n")

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*65}")
        print(f"  Odia OCR Phase 3 Monitor  —  {now}")
        print(f"{'='*65}")

        print("\n[GPUs]")
        print(gpu_stats())

        print("\n[Training Progress]")
        print(latest_loss_from_log())

        print("\n[Recent Log]")
        print(log_tail(5))

        print(f"\n[HuggingFace Hub — {args.hf_repo}]")
        print(hf_latest_commit(args.hf_repo))

        if not args.no_infer:
            print(f"\n[Quick Inference — {args.hf_repo}]")
            print(quick_inference(args.hf_repo))

        print(f"\n[Next refresh in {args.interval}s ...]")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[MONITOR] Stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
