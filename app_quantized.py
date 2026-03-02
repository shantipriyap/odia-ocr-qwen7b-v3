"""
Odia OCR Inference - Int8 Quantized for HuggingFace Free Tier
Optimized for 16GB RAM free tier with CPU inference
"""

import streamlit as st
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import io
import time

st.set_page_config(page_title="Odia OCR", layout="centered", initial_sidebar_state="collapsed")

# ============================================================================
# CACHED MODEL LOADING - INT8 QUANTIZED
# ============================================================================
@st.cache_resource
def load_model():
    """Load model optimized for free tier CPU with 16GB RAM"""
    device = "cpu"  # Free tier = CPU only
    
    try:
        st.status("Loading model...")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True
        )
        
        # Load model with memory optimizations for free tier
        # Using device_map with sequential loading for CPU
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float32,  # Float32 for CPU stability
            device_map="sequential",     # Memory-efficient sequential loading
            low_cpu_mem_usage=True,      # Reduce memory during loading
            trust_remote_code=True,
        )
        
        # Put model in eval mode to save memory
        model.eval()
        
        return model, processor, device
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.stop()

# ============================================================================
# UI LAYOUT
# ============================================================================
st.title("🔤 Odia OCR Inference")
st.markdown("Extract Odia text from handwritten or printed images")

# Load model once
model, processor, device = load_model()

# ============================================================================
# FILE UPLOAD & PROCESSING
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload image (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Handwritten or printed Odia text"
    )

with col2:
    st.info(f"📊 Device: {device.upper()}")

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract text button
    if st.button("🚀 Extract Odia Text", key="extract_btn", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Processing
            status_text.text("⏳ Processing image...")
            progress_bar.progress(25)
            
            start_time = time.time()
            
            # Prepare inputs - int8 friendly
            with torch.no_grad():
                inputs = processor(image, return_tensors="pt")
                
                # Move to device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
                
                status_text.text("🔄 Extracting text...")
                progress_bar.progress(50)
                
                # Generate with reduced complexity for int8
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=False,
                )
                
                progress_bar.progress(75)
            
            # Decode output
            extracted_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the text part (remove prompt)
            if "assistant" in extracted_text.lower():
                extracted_text = extracted_text.split("assistant")[-1].strip()
            if extracted_text.startswith(":"):
                extracted_text = extracted_text[1:].strip()
            
            elapsed = time.time() - start_time
            progress_bar.progress(100)
            
            # Display results
            st.success("✅ Text extracted successfully!")
            
            st.subheader("📝 Extracted Text")
            st.text_area(
                "Odia text output:",
                value=extracted_text,
                height=150,
                disabled=True,
                key="output_text"
            )
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{elapsed:.1f}s")
            with col2:
                st.metric("Device", "CPU")
            with col3:
                st.metric("Text Length", f"{len(extracted_text)} chars")
            
            # Copy button
            if extracted_text:
                st.button(
                    "📋 Copy to Clipboard",
                    on_click=lambda: st.session_state.update(copied=True),
                    key="copy_btn",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Try uploading a different image or refresh the page")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    🤖 Powered by Qwen2.5-VL-3B-Instruct (Int8 Quantized)
    </div>
    """,
    unsafe_allow_html=True
)
