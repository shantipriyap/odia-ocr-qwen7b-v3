#!/bin/bash
# Upload processor files to Hugging Face model repo
# Usage: bash upload_processor_files.sh

MODEL_REPO="shantipriya/hunyuan-ocr-odia"
FILES=(
  preprocessor_config.json
  merges.txt
  special_tokens_map.json
  tokenizer_config.json
  tokenizer.json
  vocab.json
)

for file in "${FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "Uploading $file..."
    huggingface-cli upload "$MODEL_REPO" "$file"
  else
    echo "File not found: $file"
  fi
done
