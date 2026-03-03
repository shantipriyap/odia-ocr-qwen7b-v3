"""
Quick inference test for shantipriya/hunyuan-ocr-odia
Diagnoses model loading issues and attempts inference.
"""
import sys
import json

print("=" * 60)
print("Testing: shantipriya/hunyuan-ocr-odia")
print("=" * 60)

# Step 1: Check transformers version
print("\n[1] Checking environment...")
try:
    import transformers
    print(f"    transformers version: {transformers.__version__}")
    import torch
    print(f"    torch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# Step 2: Check adapter_config.json base model
print("\n[2] Checking adapter_config.json for base model...")
try:
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(
        repo_id="shantipriya/hunyuan-ocr-odia",
        filename="adapter_config.json"
    )
    with open(config_path) as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg.get("base_model_name_or_path", "")
    peft_type = adapter_cfg.get("peft_type", "")
    task_type = adapter_cfg.get("task_type", "")
    print(f"    peft_type: {peft_type}")
    print(f"    task_type: {task_type}")
    print(f"    base_model_name_or_path: '{base_model}'")
    if not base_model:
        print("\n    *** CRITICAL ISSUE FOUND ***")
        print("    base_model_name_or_path is EMPTY in adapter_config.json!")
        print("    The LoRA adapter cannot be auto-loaded without a base model reference.")
        print("    You need to know which HunYuanVL model was used for training.")
except Exception as e:
    print(f"    ERROR reading adapter_config: {e}")

# Step 3: Check if HunYuanVLForConditionalGeneration is available
print("\n[3] Checking HunYuanVLForConditionalGeneration availability...")
hunyuan_available = False
try:
    from transformers import HunYuanVLForConditionalGeneration
    hunyuan_available = True
    print("    HunYuanVLForConditionalGeneration: AVAILABLE")
except ImportError:
    print(f"    NOT available in transformers {transformers.__version__}")
    print("    Minimum required: transformers >= 4.52.0 (approx)")
    # Try to find if it's added in a newer version or via trust_remote_code
    try:
        from transformers import AutoModelForVision2Seq
        print("    AutoModelForVision2Seq: AVAILABLE (may work as alternative)")
    except ImportError:
        print("    AutoModelForVision2Seq: NOT available")

# Step 4: Attempt to load processor
print("\n[4] Attempting to load processor...")
try:
    from transformers import TrOCRProcessor, AutoProcessor
    processor = TrOCRProcessor.from_pretrained("shantipriya/hunyuan-ocr-odia")
    print(f"    TrOCRProcessor loaded successfully: {type(processor)}")
except Exception as e:
    print(f"    TrOCRProcessor failed: {e}")
    try:
        processor = AutoProcessor.from_pretrained(
            "shantipriya/hunyuan-ocr-odia",
            trust_remote_code=True
        )
        print(f"    AutoProcessor loaded: {type(processor)}")
    except Exception as e2:
        print(f"    AutoProcessor also failed: {e2}")

# Step 5: Try loading model with trust_remote_code
print("\n[5] Attempting model load (with trust_remote_code)...")
try:
    from transformers import AutoModelForVision2Seq, VisionEncoderDecoderModel
    if hunyuan_available:
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            "shantipriya/hunyuan-ocr-odia",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            "shantipriya/hunyuan-ocr-odia",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
    model.eval()
    print(f"    Model loaded: {type(model)}")
    print("\n[TEST] Running a quick inference test...")
    from PIL import Image
    import requests
    from io import BytesIO
    # Fetch a simple test image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Odia_Script.jpg/640px-Odia_Script.jpg"
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        print(f"    Test image loaded: {img.size}")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\n    INFERENCE OUTPUT: {result}")
        print("\n    *** INFERENCE SUCCESSFUL ***")
    except Exception as e:
        print(f"    Inference test failed: {e}")
except Exception as e:
    print(f"    Model load failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)
print(f"transformers version: {transformers.__version__}")
print(f"HunYuanVLForConditionalGeneration available: {hunyuan_available}")
print(f"base_model_name_or_path in adapter_config: '{base_model}'")
if not base_model:
    print("\nACTION REQUIRED:")
    print("  The adapter_config.json has an empty base_model_name_or_path.")
    print("  To fix inference, update it with the base model used during training,")
    print("  e.g.: 'tencent/HunYuan-VL-A13B-Instruct' or similar.")
print("[DONE]")
