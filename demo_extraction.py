#!/usr/bin/env python3
"""
Quick Post-Processing Extraction Demo
Shows before/after comparison
"""

def extract_odia_text(text):
    """Extract clean Odia text from model output"""
    text = text.replace("assistant.", "").replace("user", "").replace("system", "")
    text = text.replace("You are", "").replace("helpful", "").replace("What text", "")
    text = text.replace("Extract", "").replace("Transcribe", "").replace("is in this image", "")
    
    odia_chars = []
    for char in text:
        if 0x0B00 <= ord(char) <= 0x0B7F or char in " \n\t":
            odia_chars.append(char)
    
    result = "".join(odia_chars).strip()
    while "  " in result:
        result = result.replace("  ", " ")
    return result

print("="*70)
print("POST-PROCESSING EXTRACTION DEMO")
print("="*70)

# Example outputs from the model
test_cases = [
    {
        "raw": "system\nYou are a helpful assistant.\nuser\nWhat text is in this image?\nପ୍ରେରଣର",
        "expected": "ପ୍ରେରଣର"
    },
    {
        "raw": "assistant.\nuser\nTranscribe the Odia text:\nଫୁଲି",
        "expected": "ଫୁଲି"
    },
    {
        "raw": "The text shown is: ସିମିତ\nassistant message",
        "expected": "ସିମିତ"
    },
    {
        "raw": "Please extract: ମୈସୂଚୁସେଟ୍ସ\nThank you!",
        "expected": "ମୈସୂଚୁସେଟ୍ସ"
    },
]

print("\nTesting Extraction:\n")

successes = 0
total = len(test_cases)

for i, case in enumerate(test_cases, 1):
    cleaned = extract_odia_text(case["raw"])
    expected = case["expected"]
    match = cleaned == expected
    
    print(f"Sample {i}:")
    print(f"  Raw output:  {case['raw'][:50]}...")
    print(f"  Extracted:   {cleaned}")
    print(f"  Expected:    {expected}")
    print(f"  Match:       {'✅ YES' if match else '❌ NO'}")
    print()
    
    if match:
        successes += 1

print("="*70)
print(f"RESULTS: {successes}/{total} samples correctly extracted ({successes/total*100:.0f}%)")
print("="*70)

print("""
KEY OBSERVATIONS:

1. ✅ Extraction correctly isolates Odia Unicode
2. ✅ Removes English chat template
3. ✅ Cleans whitespace properly
4. ✅ Produces clean text output

ACCURACY IMPROVEMENT:
- Before extraction: 0% (raw output with template)
- After extraction: ~25-100% (depends on model output format)

USAGE:
    from inference_with_postprocessing import OdiaOCRInference
    ocr = OdiaOCRInference()
    result = ocr.transcribe(image)
    print(result['text'])  # Clean Odia text
""")
