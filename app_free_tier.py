"""
Odia OCR - Free Tier Ultra-Lightweight Version
Uses TrOCR (500MB) for OCR tasks
~1.5GB memory footprint - Safe for 16GB free tier
"""

import streamlit as st
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time

st.set_page_config(page_title="Odia OCR", layout="centered", initial_sidebar_state="collapsed")

# ============================================================================
# CACHED MODEL LOADING - ULTRA LIGHT
# ============================================================================
@st.cache_resource
def load_model():
    """Load ultra-lightweight OCR model for free tier"""
    
    device = "cpu"
    
    try:
        with st.spinner("🔄 Loading model... (first time only)"):
            # TrOCR: Purpose-built OCR, much lighter than vision-language models
            # ~500MB model size vs 3.8GB for Qwen
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            model_name = "microsoft/trocr-small-handwritten"
            
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            
            return model, processor, device
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.info("Free tier has limited resources. Try a refresh in 30 seconds.")
        st.stop()

# ============================================================================
# UI LAYOUT
# ============================================================================
st.title("🔤 Odia OCR")
st.markdown("Free Tier Version - Ultra-lightweight text extraction")

try:
    model, processor, device = load_model()
except:
    st.error("Model failed to load")
    st.stop()

st.info("📊 Device: CPU | ⏱️ Speed: ~3-5s per image | 💾 Safe for free tier")

# ============================================================================
# FILE UPLOAD & PROCESSING
# ============================================================================
uploaded_file = st.file_uploader(
    "Upload image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Handwritten or printed text"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🚀 Extract Text", use_container_width=True):
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text("⏳ Processing...")
            progress.progress(30)
            
            start = time.time()
            
            # Resize for TrOCR (works best with smaller images)
            max_size = (384, 384)
            image_resized = image.copy()
            image_resized.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            with torch.no_grad():
                pixel_values = processor(images=image_resized, return_tensors="pt").pixel_values
                
                status.text("🔄 Extracting text...")
                progress.progress(60)
                
                generated_ids = model.generate(pixel_values)
                text = processor.decode(generated_ids[0], skip_special_tokens=True)
            
            elapsed = time.time() - start
            progress.progress(100)
            
            st.success("✅ Done!")
            
            st.subheader("📝 Extracted Text")
            st.text_area("Output:", value=text, height=120, disabled=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Time", f"{elapsed:.1f}s")
            with col2:
                st.metric("Length", f"{len(text)} chars")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("⚡ Free Tier Safe • TrOCR (500MB) • CPU Only • ~3-5s inference")
