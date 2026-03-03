import os
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import random
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from PIL import Image

def print_dataset_statistics(dataset, name):
    print(f"\n{'='*60}")
    print(f"📊 DATASET STATISTICS: {name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset):,}")
    # Field coverage
    field_counts = {k: 0 for k in dataset.features.keys()}
    for ex in dataset:
        for k in field_counts:
            if ex.get(k) is not None:
                field_counts[k] += 1
    for k, v in field_counts.items():
        print(f"  • {k}: {v:,} / {len(dataset):,} present")
    # Text length stats
    if 'extracted_text' in dataset.features:
        lengths = [len(str(ex.get('extracted_text',''))) for ex in dataset if ex.get('extracted_text')]
        if lengths:
            print(f"  • extracted_text length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    # Image stats
    if 'image' in dataset.features:
        img_types = {}
        for ex in dataset:
            img = ex.get('image')
            t = type(img).__name__
            img_types[t] = img_types.get(t,0)+1
        print(f"  • image types: {img_types}")
    print(f"{'='*60}\n")

def main():
    # DATASET PREP & STATISTICS
    print("\n" + "="*70)
    print("🚩 DATASET PREPARATION & STATISTICS")
    print("="*70)
    dataset_names = [
        'OdiaGenAIOCR/synthetic_data',
        'shantipriya/odia-ocr-merged',
    ]
    all_datasets = []
    for ds_name in dataset_names:
        print(f"\n📥 Loading: {ds_name}")
        ds = load_dataset(ds_name, split='train')
        print_dataset_statistics(ds, ds_name)
        all_datasets.append(ds)
    # Merge
    merged = concatenate_datasets(all_datasets)
    # Limit to first 10 samples for testing
    merged = merged.select(range(10))
    print_dataset_statistics(merged, 'MERGED (10 samples)')
    print("\n✅ DATASET READY. Starting training...\n")

    # --- TRAINING CODE STARTS HERE ---
    # Replace train_dataset with merged
    # (Below is a minimal working example, adapt as needed)
    MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct'
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    from peft import LoraConfig, get_peft_model
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    # Print device info
    device = next(model.parameters()).device
    print(f"\n🚀 Model loaded on device: {device}\n")
    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess_function(example):
        try:
            image = example.get('image')
            text = example.get('extracted_text', '')
            if image is None or not text: return None
            if isinstance(image, str):
                try: image = Image.open(image).convert('RGB')
                except: return None
            elif hasattr(image, 'convert'): image = image.convert('RGB')
            else: return None
            return {'image': image, 'text': text}
        except: return None

    print('🔄 Preprocessing merged dataset...')
    train_dataset = merged.map(preprocess_function, batched=False, num_proc=16)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    print(f'✅ {len(train_dataset):,} samples ready for training')

    class QwenOCRDataCollator:
        def __init__(self, processor): self.processor = processor
        def __call__(self, batch):
            images, texts = [], []
            for example in batch:
                try:
                    img, text = example.get('image'), example.get('text', '')
                    if img is None or not text: continue
                    if isinstance(img, str):
                        try: img = Image.open(img).convert('RGB')
                        except: continue
                    elif hasattr(img, 'convert'): img = img.convert('RGB')
                    else: continue
                    images.append(img)
                    texts.append(text)
                except: continue
            if not images: return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}
            try:
                inputs = self.processor(images, text=texts, padding=True, truncation=True, return_tensors='pt')
                inputs['labels'] = inputs['input_ids'].clone()
                return inputs
            except: return {'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([[0]])}

    from huggingface_hub import login
    login()  # Will prompt for your token if not already logged in
    import wandb
    wandb.login()  # Will prompt for your API key if not already logged in
    wandb.init(project="odia-ocr-merged", entity="shantipriya-parida")
    training_args = TrainingArguments(
        output_dir='./checkpoint-merged',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        warmup_steps=500,  # Increase for larger dataset
        logging_steps=100,  # Log less frequently for large dataset
        save_strategy='epoch',
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        remove_unused_columns=False,
        dataloader_num_workers=8,
        optim='adamw_torch',
        report_to=["tensorboard", "wandb", "csv"],
        run_name="odia-ocr-merged-wandb",
        evaluation_strategy='no',
        bf16=True,
        seed=42,
        push_to_hub=True,
        hub_model_id='shantipriya/odia-ocr-merged',
        hub_strategy='every_save',
        logging_dir='./logs',
    )

    # Callback to save loss/accuracy for plotting
    import csv
    class LossLoggerCallback:
        def __init__(self, filename='training_metrics.csv'):
            self.filename = filename
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'loss'])
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([state.global_step, logs['loss']])
                # Print log to console in real time
                print(f"[Step {state.global_step}] loss: {logs['loss']}", flush=True)

    loss_logger = LossLoggerCallback()

    print('🎯 Creating trainer...')
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=QwenOCRDataCollator(processor), callbacks=[loss_logger])
    print('✅ Ready!')
    print('='*70)
    print('🚀 STARTING TRAINING')
    print('='*70)

    print('🎯 Creating trainer...')
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=QwenOCRDataCollator(processor))
    print('✅ Ready!')
    print('='*70)
    print('🚀 STARTING TRAINING')
    print('='*70)
    try:
        trainer.train()
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    print('\n' + '='*70)
    print('✅ TRAINING COMPLETE')
    print('='*70)
if __name__ == "__main__":
lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
def preprocess_function(example):
    main()
