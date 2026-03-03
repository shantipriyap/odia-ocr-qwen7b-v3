#!/usr/bin/env python3
"""
Push fine-tuned Qwen2.5-VL model to HuggingFace Hub
Run on A100 server: ssh root@95.216.229.232 "cd /root/odia_ocr && HF_TOKEN='your_token' python3 hf_push_model.py"
"""

import os
import sys

def main():
    print("="*70)
    print("PUSHING ODIA OCR MODEL TO HUGGINGFACE HUB")
    print("="*70)
    
    # Get token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n❌ HF_TOKEN not set in environment")
        print("Usage: HF_TOKEN='your_token' python3 hf_push_model.py")
        sys.exit(1)
    
    # Import after checking token
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from huggingface_hub import login
    
    try:
        print("\n[1/3] Authenticating with HuggingFace...")
        login(token=token)
        print("      ✅ Authenticated")
        
        print("\n[2/3] Loading model and processor...")
        checkpoint_path = "./checkpoint-qwen-full-fixed/checkpoint-3500"
        base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        print("      ✅ Model & processor loaded (8.8GB checkpoint)")
        
        print("\n[3/3] Pushing to HuggingFace Hub...")
        repo_id = "shantipriya/odia-ocr-qwen-finetuned"
        
        print(f"      → Model URL: https://huggingface.co/{repo_id}")
        model.push_to_hub(
            repo_id,
            token=token,
            private=False,
            commit_message="Fine-tuned Qwen2.5-VL-3B for Odia OCR - Loss: 5.5→0.09 (98% improvement)"
        )
        print("      ✅ Model pushed")
        
        processor.push_to_hub(
            repo_id,
            token=token,
            private=False,
            commit_message="AutoProcessor for Odia OCR model"
        )
        print("      ✅ Processor pushed")
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"\n📦 Model Repository:")
        print(f"   https://huggingface.co/{repo_id}")
        print(f"\n💡 Usage with inference wrapper:")
        print(f"   from inference_with_postprocessing import OdiaOCRInference")
        print(f"   ocr = OdiaOCRInference(model_path='{repo_id}')")
        print(f"   result = ocr.transcribe(image)")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
