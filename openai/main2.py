import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai/gpt-oss-20b"

# 1) Load with trust_remote_code so custom classes are allowed
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",      # or torch.float16 / bfloat16 explicitly
    device_map="auto"        # put it on your GPU if available
)

    
# Define a prompt
prompt = "def fibonacci(n):"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(

    inputs["input_ids"],
    max_new_tokens=1000,   # number of tokens to generate
    temperature=0.7,      # control randomness (lower = less random)
    do_sample=True,       # set True for sampling-based generation
    top_p=0.95,           # nucleus sampling, for diverse yet coherent output
)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
