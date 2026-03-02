import requests
import time

space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
api_url = f"https://huggingface.co/api/spaces/{space_id}"

print("=" * 70)
print("🔍 CHECKING SPACE STATUS")
print("=" * 70)

for attempt in range(1, 13):
    try:
        response = requests.get(api_url, timeout=5)
        data = response.json()
        status = data.get('runtime', {}).get('stage', 'unknown')
        
        elapsed = (attempt - 1) * 5
        print(f"[{elapsed:02d}s] Status: {status}")
        
        if status == 'RUNNING':
            print("\n" + "=" * 70)
            print("✅✅✅ SPACE IS LIVE! ✅✅✅")
            print("=" * 70)
            print(f"🎉 Your Odia OCR app is ready!")
            print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
            print("=" * 70)
            break
    except Exception as e:
        print(f"[{elapsed:02d}s] checking...")
    
    if attempt < 12:
        time.sleep(5)
    
    if attempt == 12 and status != 'RUNNING':
        print(f"\n⏳ Still building (Status: {status})")
        print("Check back in 1-2 minutes")
