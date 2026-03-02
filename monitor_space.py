#!/usr/bin/env python3
"""Monitor HF Space build status"""
import requests
import time
import sys

space_url = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
api_url = f"https://huggingface.co/api/spaces/{space_url}"

print("\n" + "="*70)
print("🔍 LIVE SPACE MONITOR")
print("="*70)

for i in range(24):
    try:
        resp = requests.get(api_url, timeout=5)
        data = resp.json()
        stage = data.get('runtime', {}).get('stage', '?')
        elapsed = i * 5
        min_e, sec_e = elapsed // 60, elapsed % 60
        
        print(f"[{min_e:02d}:{sec_e:02d}] {stage}")
        
        if stage == 'RUNNING':
            print("\n✅ SPACE IS LIVE!")
            print("="*70)
            print("🎉 Your OCR app is ready!")
            print(f"https://huggingface.co/spaces/{space_url}")
            print("="*70 + "\n")
            sys.exit(0)
        elif stage == 'ERROR':
            print("\n❌ BUILD FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"[??] Error: {str(e)[:40]}")
    
    if i < 23:
        time.sleep(5)

print("\n⏳ Still building...")
print(f"Check: https://huggingface.co/spaces/{space_url}\n")
