# ===============================
# Helper: Build Prompt
# ===============================
def build_prompt(text):
    return f"Translate the following sentence to Gujarati:\n{text}\nGujarati:"
# ===============================
# Helper: Chunk Text
# ===============================
def chunk_text(text, tokenizer):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_length:
        return [text]
    chunks = []
    for i in range(0, len(tokens), max_length):
        part = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(part))
    return chunks
# ===============================
# Hardcoded Parameters
# ===============================
input_file = "input.txt"
output_file = "output.txt"
model_name = "sarvamai/sarvam-translate"
max_length = 512
dtype = "bfloat16"
gpu_memory_util = 0.85

# ===============================
# Tokenizer Initialization
# ===============================
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ===============================
# Environment Setup for H100
# ===============================
os.environ["TORCH_COMPILE"] = "0"                  # disables torch.compile
os.environ["TORCHINDUCTOR_DISABLE"] = "1"          # disables torch_inductor backend
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"     # disables Triton flash attention
os.environ["CUDA_MODULE_LOADING"] = "LAZY"         # optional, improves H100 loading


# ===============================
# Hardcoded Parameters
# ===============================
input_file = "input.txt"
output_file = "output.txt"
model_name = "sarvamai/sarvam-translate"
max_length = 512
dtype = "bfloat16"
gpu_memory_util = 0.85

# ===============================
# Initialize LLM
# ===============================
llm = LLM(
    model=model_name,
    dtype=dtype,                  # FP16 or BF16 for H100
    tensor_parallel_size=1,
    gpu_memory_utilization=gpu_memory_util,
    trust_remote_code=True,
    max_model_len=max_length,
)

# ===============================
# Prepare Input
# ===============================

def translate_column(texts, tokenizer, llm, sampling_params):
    all_outputs = []
    for idx, text in enumerate(tqdm(texts, desc="Translating", unit="item"), 1):
        chunks = chunk_text(text, tokenizer)
        prompts = [build_prompt(c) for c in chunks]
        outputs = llm.generate(prompts, sampling_params)
        merged = " ".join(o.outputs[0].text.strip() for o in outputs)
        print(f"\n---\nINPUT: {text}\nOUTPUT: {merged}\n---\n", flush=True)
        all_outputs.append(merged)
    return all_outputs

if __name__ == "__main__":
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    translated_lines = translate_column(lines, tokenizer, llm, SamplingParams(max_tokens=max_length))
    with open(output_file, "w", encoding="utf-8") as f:
        for t_line in translated_lines:
            f.write(t_line + "\n")
    print(f"Translation completed! Output saved to {output_file}")

print(f"Translation completed! Output saved to {output_file}")