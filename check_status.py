import requests
import time

space_id = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
api_url = f"https://huggingface.co/api/spaces/{space_id}"

print("=" * 70)
print("🔍 MONITORING SPACE BUILD")
print("=" * 70)

for attempt in range(1, 6):
    try:
        response = requests.get(api_url, timeout=5)
        data = response.json()
        status = data.get('runtime', {}).get('stage', 'unknown')
        
        print(f"[Attempt {attempt}] Status: {status}")
        
        if status == 'RUNNING':
            print("\n" + "=" * 70)
            print("✅ SPACE IS LIVE!")
            print("=" * 70)
            print(f"🎉 Your app is ready to use!")
            print(f"📍 URL: https://huggingface.co/spaces/{space_id}")
            print("=" * 70)
            break
    except Exception as e:
        print(f"   (checking...)")
    
    if attempt < 5:
        time.sleep(3)
