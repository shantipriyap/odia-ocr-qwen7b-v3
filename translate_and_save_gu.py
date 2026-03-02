from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Environment setup (optional, for H100)
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# Hardcoded parameters
input_file = "input.txt"
output_file = "output.txt"
hf_dir = "Reasoning_GU"
model_name = "sarvamai/sarvam-translate"
max_length = 512
dtype = "bfloat16"
gpu_memory_util = 0.85

def chunk_text(text, tokenizer):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_length:
        return [text]
    chunks = []
    for i in range(0, len(tokens), max_length):
        part = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(part))
    return chunks

def build_prompt(text):
    return f"Translate the following sentence to Gujarati:\n{text}\nGujarati:"

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

def main():
    # Initialize tokenizer and LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_util,
        trust_remote_code=True,
        max_model_len=max_length,
    )
    sampling_params = SamplingParams(max_tokens=max_length)

    # Read input
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Translate
    translated_lines = translate_column(lines, tokenizer, llm, sampling_params)

    # Save output.txt
    with open(output_file, "w", encoding="utf-8") as f:
        for t_line in translated_lines:
            f.write(t_line + "\n")
    print(f"Translation completed! Output saved to {output_file}")

    # Save Hugging Face dataset
    records = [{"input": inp, "output": out} for inp, out in zip(lines, translated_lines)]
    hf_ds = Dataset.from_list(records)
    hf_ds.save_to_disk(hf_dir)
    print(f"Saved Hugging Face dataset to {hf_dir}/")

if __name__ == "__main__":
    main()
