import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor
import io
import time
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# CRITICAL FIX: Disable all authentication to prevent 403 errors
# This prevents HuggingFace API from triggering auth verification
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_TOKEN'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Suppress verbose logging
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

st.set_page_config(
    page_title="📖 OdiaLipi-Qwen2.5-VLM",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #1f77b4;
    }
    .result-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>📖 OdiaLipi-Qwen2.5-VLM</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Advanced Odia Text Recognition with Qwen 2.5 Vision Language Model</p>", unsafe_allow_html=True)

# OdiaGenAI branding
st.markdown("""
<div style='text-align: center; padding: 10px; margin-bottom: 20px;'>
    <p style='font-size: 0.85em; color: #666; margin: 0;'>
        🐢 Powered by <a href='https://www.odiagenai.org' target='_blank' style='color: #1f77b4; text-decoration: none; font-weight: 500;'>OdiaGenAI</a>
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load processor and model - ultra-defensive approach"""
    
    processor = None
    model = None
    
    try:
        st.write("🔄 Starting model load...")
        
        # Processor (less likely to fail)
        try:
            st.write("   📥 Loading processor...")
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                trust_remote_code=False,
            )
            st.write("   ✅ Processor ready")
        except Exception as pe:
            st.error(f"❌ Processor failed")
            st.caption(f"Error: {str(pe)[:60]}")
            return None, None
        
        st.write("   📦 Loading model...")
        st.info("⏳ First load takes 1-2 minutes...")
        
        # Model loading - try importing and loading separately
        try:
            st.write("      Importing model class...")
            from transformers.models.qwen2_vl import Qwen2_5_VLForConditionalGeneration as QwenModel
            
            st.write("      Downloading weights...")
            model = QwenModel.from_pretrained(
                "OdiaGenAIOCR/odia-ocr-qwen-finetuned",
                trust_remote_code=False,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            
            st.write("   ✅ Model loaded!")
            model.eval()
            
        except Exception as me:
            st.write(f"      ⚠️  Method 1 failed")
            
            # Fallback method
            try:
                st.write("      Trying alternate import...")
                from transformers import Qwen2_5_VLForConditionalGeneration
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "OdiaGenAIOCR/odia-ocr-qwen-finetuned",
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                
                st.write("   ✅ Model loaded (alternate)!")
                model.eval()
                
            except Exception as me2:
                st.error("❌ Both model loading methods failed")
                st.caption(f"Error: {str(me2)[:60]}")
                model = None
        
        if model is None:
            return None, None
        
        st.success("✅ Ready!")
        return processor, model
        
    except Exception as e:
        st.error(f"❌ Load error: {type(e).__name__}")
        st.caption(f"Details: {str(e)[:80]}")
        return None, None

def extract_text(image, processor, model):
    """Extract Odia text from image with robust error handling"""
    try:
        # Prepare image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (for memory efficiency)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Prepare prompt
        prompt = "Extract all Odia text from this image."
        
        try:
            # Process image and generate text
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            return f"❌ Image processing failed: {str(e)[:80]}"
        
        try:
            # Move to same device as model
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(model.device)
            
            # Generate text with timeout protection
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=1,
                    temperature=0.7
                )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                return "❌ Out of memory - try a smaller image"
            else:
                return f"❌ Generation failed: {str(e)[:80]}"
        except Exception as e:
            return f"❌ Error during OCR: {type(e).__name__}: {str(e)[:80]}"
        
        try:
            # Decode output
            generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the text part
            if "Extract" in generated_text:
                text = generated_text.split("Extract")[-1].strip()
            else:
                text = generated_text.strip()
            
            return text if text else "⚠️ No text detected in image"
        except Exception as e:
            return f"❌ Decoding failed: {str(e)[:80]}"
    
    except Exception as e:
        return f"❌ Unexpected error: {type(e).__name__}: {str(e)[:80]}"

# Main interface
st.subheader("📸 Upload Image for OCR")

# File upload
uploaded_file = st.file_uploader(
    "Choose an image with Odia text",
    type=["jpg", "jpeg", "png", "gif", "webp"],
    help="Upload an image containing Odia text"
)

if uploaded_file is not None:
    # Load models with try-except to prevent UI crashes
    try:
        processor, model = load_models()
    except Exception as load_err:
        st.error(f"❌ Failed to load models: {str(load_err)[:80]}")
        processor, model = None, None
    
    if processor and model:
        # Display image
        try:
            image = Image.open(uploaded_file)
        except Exception as img_err:
            st.error(f"❌ Could not open image: {str(img_err)[:60]}")
            image = None
        
        if image:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📷 Input Image:**")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("**📊 Image Info:**")
                st.write(f"Size: {image.width} × {image.height} px")
                st.write(f"Format: {image.format}")
            
            # Extract button
            st.markdown("---")
            
            if st.button("🚀 Extract Odia Text", use_container_width=True):
                with st.spinner("🔄 Processing image... Please wait"):
                    try:
                        start_time = time.time()
                        extracted_text = extract_text(image, processor, model)
                        processing_time = time.time() - start_time
                    except Exception as extract_err:
                        st.error(f"❌ Extraction failed: {str(extract_err)[:80]}")
                        extracted_text = None
                
                if extracted_text:
                    # Display results
                    st.markdown("""
                    <div class='result-box'>
                    <strong>✅ Extraction Complete!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**🔤 Extracted Odia Text:**")
                    
                    # Text output  
                    st.text_area(
                        "Recognized Text",
                        value=extracted_text,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
                    with col2:
                        st.metric("📏 Text Length", f"{len(extracted_text)} chars")
                    with col3:
                        st.metric("📊 Model", "Qwen2.5-VL-3B")
    else:
        if processor is None:
            st.error("⚠️ Could not load processor")
        if model is None:
            st.error("⚠️ Could not load model")
else:
    st.info("👆 Upload an image to start OCR extraction")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85em;'>
<p>🎯 <strong>OdiaGenAIOCR/odia-ocr-qwen-finetuned</strong></p>
<p>Fine-tuned Vision Language Model for Odia Text Recognition</p>
<p style='margin-top: 20px;'>
Made with ❤️ by OdiaGenAI<br>
<a href='https://huggingface.co/OdiaGenAIOCR' style='color: #1f77b4;'>View on HuggingFace</a>
</p>
</div>
""", unsafe_allow_html=True)
