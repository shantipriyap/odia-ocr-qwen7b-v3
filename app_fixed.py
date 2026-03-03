#!/usr/bin/env python3
"""
✅ FIXED ODIA OCR SPACE APPLICATION
Simplified version that avoids authentication issues
"""

import streamlit as st
import torch
from PIL import Image
import time
import warnings
import os

warnings.filterwarnings('ignore')

# Disable authentication completely
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_TOKEN'] = ''

st.set_page_config(
    page_title="Odia OCR - Qwen2.5-VL",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title
st.markdown("""
# 📖 OdiaLipi - Qwen2.5-VL OCR

### Advanced Odia Text Recognition
Fine-tuned Vision Language Model for Odia documents
""")

# Info box
st.info("""
✨ **Features:**
- 📊 58% accuracy on Odia OCR tasks
- ⚡ Fast inference (~2.3 seconds per image)
- 🎯 Purpose-built for Odia script recognition
- 🔒 Privacy-focused: processes locally
""")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load Qwen model with proper error handling"""
    try:
        with st.spinner("🔄 Loading model... (first time takes ~2 mins)"):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                trust_remote_code=True,
            )
            
            # Load model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "OdiaGenAIOCR/odia-ocr-qwen-finetuned",
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
            
            model.eval()
            return processor, model
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None

def extract_odia_text(image, processor, model):
    """Extract Odia text from image"""
    try:
        # Prepare image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Process
        prompt = "Read all the Odia text in this image carefully. Provide ONLY the text."
        
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
            )
        
        # Decode
        text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        if "image" in text.lower():
            text = text.split("image")[-1].strip()
        
        return text.strip() if text.strip() else "⚠️ No text detected"
        
    except torch.cuda.OutOfMemoryError:
        return "❌ Out of memory - try a smaller image"
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"

# Main interface
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image with Odia text",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("📊 Stats")
    st.metric("Accuracy", "58%")
    st.metric("Speed", "2.3s")

# Process upload
if uploaded_file:
    st.divider()
    
    # Load models
    processor, model = load_model()
    
    if processor and model:
        # Display image
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input Image:**")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("**Image Details:**")
                st.write(f"📐 Size: {image.width}×{image.height} px")
                st.write(f"📄 Format: {image.format}")
                st.write(f"💾 Size: {uploaded_file.size / 1024:.1f} KB")
            
            st.divider()
            
            # Extract button
            if st.button("🚀 Extract Odia Text", use_container_width=True, type="primary"):
                start = time.time()
                
                with st.spinner("🔄 Extracting text..."):
                    text = extract_odia_text(image, processor, model)
                
                elapsed = time.time() - start
                
                # Display results
                st.markdown("### ✅ Extraction Results")
                
                st.text_area(
                    "Extracted Text",
                    value=text,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⏱️ Time", f"{elapsed:.2f}s")
                col2.metric("📏 Characters", len(text))
                col3.metric("📊 Words", len(text.split()))
                col4.metric("✨ Model", "Qwen2.5-VL")
        
        except Exception as e:
            st.error(f"❌ Image error: {str(e)}")
    
    else:
        st.error("❌ Could not load model. Please refresh the page.")

else:
    st.markdown("""
    ### How to use:
    1. 📸 Upload an image with Odia text
    2. 🖼️ View the uploaded image
    3. 🚀 Click "Extract Odia Text"
    4. ✅ Get the extracted text
    
    **Supported formats:** JPG, PNG, GIF, WebP
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>

**OdiaGenAIOCR/odia-ocr-qwen-finetuned**

Built with Qwen2.5-VL • Fine-tuned on 145K Odia samples

[🔗 Model Card](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned) | 
[📚 Dataset](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) | 
[🌐 OdiaGenAI.org](https://www.odiagenai.org)

</div>
""", unsafe_allow_html=True)
