import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")

# Ensure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    
# Define a prompt
prompt = "def fibonacci(n):"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(

    inputs["input_ids"],
    max_new_tokens=100,   # number of tokens to generate
    temperature=0.7,      # control randomness (lower = less random)
    do_sample=True,       # set True for sampling-based generation
    top_p=0.95,           # nucleus sampling, for diverse yet coherent output
)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
