import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "openai/gpt-oss-20b"  # your repo
# If you’re on GPU and the repo ships 4-bit weights, keep this; otherwise set bnb_config=None.
bnb_config = None  # BitsAndBytesConfig(load_in_4bit=True)  # only if the model actually supports it

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True   # important for custom chat templates/tokenization
)

# Device / dtype: pick ONE of these blocks

# A) CPU, safest
device = "cpu"
dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# # B) GPU, bf16 (preferred on Ampere+)
# device = "cuda"
# dtype = torch.bfloat16
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=dtype,
#     device_map="auto",
#     trust_remote_code=True,
#     quantization_config=bnb_config   # only if the model provides 4-bit weights
# )

# Build a proper chat prompt using the model’s template
messages = [
    {"role": "system", "content": "You are a helpful code review assistant."},
    {"role": "user", "content": "Review this Python function for clarity and bugs:\n\n"
                                "def foo(x):\n    if x==0:\n        return 1\n    return x*foo(x-1)"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True      # appends the model’s expected assistant prefix
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Make sure eos/pad are sane
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

gen_kwargs = dict(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,   # Playground often uses 0.6–0.8; drop to 0–0.2 for deterministic
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

with torch.no_grad():
    out = model.generate(**inputs, **gen_kwargs)

text = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(text)
