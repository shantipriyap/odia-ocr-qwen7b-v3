from transformers import TrOCRProcessor

# Download and save the processor files from microsoft/trocr-base-handwritten
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
processor.save_pretrained('./trocr_processor')
print("Processor files saved to ./trocr_processor")
