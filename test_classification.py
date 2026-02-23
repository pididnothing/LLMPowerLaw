"""
Quick test script to verify local model classification
Tests TinyLlama with proper prompting
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading TinyLlama...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Test classification prompt
test_prompt = """Task: Determine if two questions are paraphrases (asking the same thing).
Output only "1" if they are paraphrases, or "0" if they are not.

Question 1: Why are African-Americans so beautiful?
Question 2: Why are hispanics so beautiful?

Answer (0 or 1):"""

print("\n" + "="*60)
print("Testing Classification")
print("="*60)
print(f"\nPrompt:\n{test_prompt}")

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    temperature=0.1,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract just the new generation
new_text = response[len(test_prompt):].strip()
print(f"\nModel Response: '{new_text}'")

# Try to extract label
try:
    # Look for first digit
    for char in new_text:
        if char in ['0', '1']:
            prediction = char
            print(f"Extracted Label: {prediction}")
            break
    else:
        print("Could not extract 0 or 1 from response")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)

# Test a few more examples
test_cases = [
    ("Is there a reason why we should travel alone?", 
     "What are some reasons to travel alone?", 
     "1"),  # Should be paraphrase
    
    ("Why are people so obsessed with having a girlfriend/boyfriend?", 
     "How can a single male have a child?", 
     "0"),  # Should NOT be paraphrase
]

print("\nTesting Multiple Examples:")
print("="*60)

for q1, q2, expected in test_cases:
    prompt = f"""Task: Are these questions asking the same thing?
Question 1: {q1}
Question 2: {q2}
Answer (0=No, 1=Yes):"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = response[len(prompt):].strip()
    
    # Extract prediction
    pred = None
    for char in new_text.split()[0] if new_text else "":
        if char in ['0', '1']:
            pred = char
            break
    
    result = "✓" if pred == expected else "✗"
    print(f"\n{result} Expected: {expected}, Got: {pred}")
    print(f"   Q1: {q1[:50]}...")
    print(f"   Q2: {q2[:50]}...")
    print(f"   Response: {new_text[:50]}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
