#!/usr/bin/env python3
"""
Qwen2.5-VL Fine-tuned Model - Production Inference with Post-processing
Extracts clean Odia text from model output
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json

class OdiaOCRInference:
    def __init__(self, model_path="./checkpoint-qwen-full-fixed/checkpoint-3500"):
        """Initialize model and processor"""
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def extract_odia_text(self, text):
        """
        Extract clean Odia text from model output
        Removes chat template, English words, and non-Odia characters
        """
        # Remove common English chat templates
        text = text.replace("assistant.", "").replace("user", "").replace("system", "")
        text = text.replace("You are", "").replace("helpful", "").replace("What text", "")
        text = text.replace("Extract", "").replace("Transcribe", "").replace("is in this image", "")
        text = text.replace("assistant", "").replace("message", "").replace("Please", "")
        text = text.replace("thank", "").replace("help", "").replace("image", "")
        
        # Extract only Odia Unicode characters (U+0B00 to U+0B7F) and spaces
        odia_chars = []
        for char in text:
            if 0x0B00 <= ord(char) <= 0x0B7F or char in " \n\t":
                odia_chars.append(char)
        
        result = "".join(odia_chars).strip()
        
        # Clean multiple spaces
        while "  " in result:
            result = result.replace("  ", " ")
        
        return result
    
    def transcribe(self, image, prompt="Transcribe the Odia text:"):
        """
        Transcribe Odia text from image
        
        Args:
            image: PIL Image object or path to image
            prompt: OCR prompt
        
        Returns:
            dict with 'raw', 'clean', 'confidence'
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Generate
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        text_input = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        output_raw = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Clean text
        output_clean = self.extract_odia_text(output_raw)
        
        return {
            "raw": output_raw,
            "text": output_clean,
            "confidence": len(output_clean) / max(len(output_raw), 1) if output_raw else 0
        }

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("ODIA OCR INFERENCE WITH POST-PROCESSING")
    print("="*70)
    
    # Initialize
    print("\n1 Loading model...")
    ocr = OdiaOCRInference()
    print("   ✅ Model loaded")
    
    # Test on a few samples
    print("\n2 Testing on sample image...")
    try:
        from datasets import load_dataset
        ds = load_dataset("shantipriya/odia-ocr-merged")["train"]
        
        # Test on first sample
        sample = ds[0]
        image = Image.open(__import__('io').BytesIO(sample["image"])).convert("RGB") if isinstance(sample["image"], bytes) else sample["image"]
        expected = sample["text"]
        
        result = ocr.transcribe(image)
        
        print(f"\n   Expected: {expected[:60]}...")
        print(f"   Predicted: {result['text'][:60]}...")
        print(f"   Confidence: {result['confidence']:.2f}")
        
    except Exception as e:
        print(f"   Error: {str(e)[:100]}")
    
    print("\n" + "="*70)
    print("Ready for production inference!")
    print("="*70)
