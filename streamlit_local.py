#!/usr/bin/env python3
"""Local Streamlit Odia OCR using HF Inference API."""
import base64
import io
import os
import requests
import streamlit as st
from PIL import Image
import sys

# Ensure stdout/stderr accept Odia Unicode output to avoid ASCII codec errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

API_URL = "https://router.huggingface.co/v1/chat/completions"
CUSTOM_API_URL = os.environ.get("HF_OCR_ENDPOINT_URL")  # Optional direct endpoint URL
HF_INFERENCE_URL_TMPL = "https://api-inference.huggingface.co/models/{model}"
# Default to the published finetuned model; allow overriding via env
FINETUNED_MODEL_ID = "OdiaGenAIOCR/odia-ocr-qwen-finetuned"
BASE_MODEL_ID = os.environ.get("HF_OCR_BASE_MODEL") or "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_ID = os.environ.get("HF_OCR_ADAPTER") or FINETUNED_MODEL_ID
# Model actually sent to the router; prefer base + adapter for chat compatibility
MODEL_ID = os.environ.get("HF_OCR_MODEL") or BASE_MODEL_ID
# Allow multiple env var names for convenience
_env_order = ["HF_OCR_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]
HF_TOKEN = None
HF_TOKEN_SOURCE = None
for _env in _env_order:
    if os.environ.get(_env):
        HF_TOKEN = os.environ.get(_env)
        HF_TOKEN_SOURCE = _env
        break

# Allow user-provided token in the UI when env vars are absent
USER_TOKEN_KEY = "user_supplied_token"

st.set_page_config(page_title="Odia OCR (HF Inference)", layout="wide")
st.title("Odia OCR (HF Inference)")
st.markdown("Upload an image, then click **Extract Text**. Uses HF Inference for Qwen2.5-VL.")

with st.sidebar:
    st.subheader("Settings")
    st.write(f"Model: {MODEL_ID}")
    if MODEL_ID == BASE_MODEL_ID and ADAPTER_ID:
        st.write(f"Adapter: {ADAPTER_ID}")
    user_token = st.text_input(
        "HF token (if not set in env)",
        value=st.session_state.get(USER_TOKEN_KEY, ""),
        type="password",
        help="Paste a Hugging Face token with Inference access.",
    )
    if user_token:
        st.session_state[USER_TOKEN_KEY] = user_token.strip()
    effective_token = HF_TOKEN or st.session_state.get(USER_TOKEN_KEY)
    if effective_token:
        st.success(f"Token in use via {'env ' + HF_TOKEN_SOURCE if HF_TOKEN else 'sidebar input'}")
    else:
        st.error("Provide a token via env or sidebar to call HF Inference.")
    max_tokens = st.slider("Max tokens", 64, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
extract_btn = st.button("Extract Text", type="primary")
output_area = st.empty()


def call_inference(img: Image.Image, max_tokens: int, temperature: float):
    effective_token = HF_TOKEN or st.session_state.get(USER_TOKEN_KEY)
    if not effective_token:
        return "ERROR: Missing HF token (set env or use sidebar input)"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    effective_model = MODEL_ID
    parameters = {}
    if MODEL_ID == FINETUNED_MODEL_ID:
        # Finetuned repo is not chat; route through base with adapter
        effective_model = BASE_MODEL_ID
        parameters = {"adapter_id": FINETUNED_MODEL_ID}
    elif ADAPTER_ID and MODEL_ID == BASE_MODEL_ID:
        parameters = {"adapter_id": ADAPTER_ID}
    payload = {
        "model": effective_model,
        "parameters": parameters if parameters else None,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all Odia text from this image. Return only the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not parameters:
        payload.pop("parameters", None)
    api_url = CUSTOM_API_URL or API_URL
    # First try chat/completions via router
    try:
        resp = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {effective_token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return "Warning: No output"
            content = choices[0].get("message", {}).get("content", "")
            return content.strip() if content else "Warning: Empty response"
        # If router rejects model/provider, fall back to direct inference endpoint
        if api_url == API_URL and resp.status_code in (400, 404, 410):
            fallback_url = HF_INFERENCE_URL_TMPL.format(model=effective_model)
            inf_payload = {
                "inputs": {
                    "image": f"data:image/png;base64,{img_b64}",
                    "text": "Extract all Odia text from this image. Return only the text.",
                },
                "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
            }
            if parameters.get("adapter_id"):
                inf_payload["parameters"]["adapter_id"] = parameters["adapter_id"]
            inf_resp = requests.post(
                fallback_url,
                headers={"Authorization": f"Bearer {effective_token}"},
                json=inf_payload,
                timeout=60,
            )
            if inf_resp.status_code == 200:
                try:
                    data = inf_resp.json()
                    # HF inference may return list/dict; extract text heuristically
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        generated = data[0].get("generated_text") or data[0].get("generated_texts")
                        if generated:
                            return generated if isinstance(generated, str) else "\n".join(generated)
                    if isinstance(data, dict):
                        gen = data.get("generated_text") or data.get("text")
                        if gen:
                            return gen if isinstance(gen, str) else "\n".join(gen)
                    return str(data)[:400]
                except Exception:
                    return inf_resp.text[:400]
            return f"ERROR: Inference {inf_resp.status_code}: {inf_resp.text[:200]}"
        return f"ERROR: API {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"ERROR: Request failed: {str(e)[:200]}"


if extract_btn:
    if not uploaded:
        output_area.error("Please upload an image first.")
    else:
        try:
            img = Image.open(uploaded)
            with st.spinner("Calling HF Inference..."):
                text = call_inference(img, max_tokens=max_tokens, temperature=temperature)
            output_area.text_area("Extracted Text", text, height=240)
        except Exception as e:
            output_area.error(f"Failed to process image: {e}")

st.markdown("---")
st.markdown(
    "If you see token errors, either set an env var (HF_OCR_TOKEN / HF_TOKEN / HUGGINGFACEHUB_API_TOKEN) before launch, or paste the token in the sidebar."
)
