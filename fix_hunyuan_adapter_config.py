"""
Fix adapter_config.json for shantipriya/hunyuan-ocr-odia
Sets the missing base_model_name_or_path to Qwen/Qwen2.5-VL
"""
import json
from huggingface_hub import hf_hub_download, HfApi

HF_REPO = "shantipriya/hunyuan-ocr-odia"
BASE_MODEL = "Qwen/Qwen2.5-VL"
HF_TOKEN = "os.getenv("HF_TOKEN", "")"

print(f"Downloading adapter_config.json from {HF_REPO}...")
config_path = hf_hub_download(
    repo_id=HF_REPO,
    filename="adapter_config.json",
    token=HF_TOKEN
)

with open(config_path) as f:
    adapter_cfg = json.load(f)

print(f"Current base_model_name_or_path: '{adapter_cfg.get('base_model_name_or_path', '')}'")

# Fix the base model
adapter_cfg["base_model_name_or_path"] = BASE_MODEL

# Save fixed config locally
fixed_path = "/tmp/adapter_config_fixed.json"
with open(fixed_path, "w") as f:
    json.dump(adapter_cfg, f, indent=2)

print(f"Fixed config saved to {fixed_path}")
print(f"New base_model_name_or_path: '{adapter_cfg['base_model_name_or_path']}'")

# Upload back to HF
print(f"\nUploading fixed adapter_config.json to {HF_REPO}...")
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=fixed_path,
    path_in_repo="adapter_config.json",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Fix: set base_model_name_or_path to Qwen/Qwen2.5-VL"
)
print("Done! adapter_config.json updated on Hugging Face.")
print(f"\nYou can now load the model with:")
print(f"  from peft import PeftModel")
print(f"  from transformers import AutoModelForVision2Seq, AutoProcessor")
print(f"  base = AutoModelForVision2Seq.from_pretrained('{BASE_MODEL}', ...)")
print(f"  model = PeftModel.from_pretrained(base, '{HF_REPO}')")
