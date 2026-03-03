import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Example usage:
# python vllm_inference_multi_column.py \
#   --model ai4bharat/IndicTrans3-beta \
#   --input_file input.jsonl \
#   --output_file output.jsonl \
#   --src_lang en \
#   --tgt_lang gu \
#   --columns user_input reasoning target_language

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--src_lang', required=True)
    parser.add_argument('--tgt_lang', required=True)
    parser.add_argument('--columns', nargs='+', required=True, help='Columns to translate')
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def build_prompt(text, src_lang, tgt_lang):
    return f"{src_lang} to {tgt_lang}: {text}"

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=256)

    # Read input JSONL
    with open(args.input_file, 'r', encoding='utf-8') as fin:
        records = [json.loads(line) for line in fin]

    # For each column, translate and add new column with _gu suffix
    for col in args.columns:
        texts = [r.get(col, '') for r in records]
        prompts = [build_prompt(t, args.src_lang, args.tgt_lang) if t else '' for t in texts]
        outputs = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f'Translating {col}'):
            batch_prompts = prompts[i:i+args.batch_size]
            batch_outputs = [''] * len(batch_prompts)
            if any(batch_prompts):
                results = llm.generate(batch_prompts, sampling_params)
                batch_outputs = [o.outputs[0].text.strip() if o.outputs else '' for o in results]
            outputs.extend(batch_outputs)
        # Add translated column
        for r, out in zip(records, outputs):
            r[f'{col}_gu'] = out

    # Write output JSONL (same format as input, with extra columns)
    with open(args.output_file, 'w', encoding='utf-8') as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Wrote {len(records)} records to {args.output_file}")

if __name__ == '__main__':
    main()
