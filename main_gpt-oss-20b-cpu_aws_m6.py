import torch,os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#model_id = "openai/gpt-oss-20b"  # your repo
# os.environ["HF_HOME"] = "/models"
# os.environ["TRANSFORMERS_CACHE"] = "/models/hf/transformers"

model_path = r"/mount/gpt-oss-20b-dequant-bf16"
# If you’re on GPU and the repo ships 4-bit weights, keep this; otherwise set bnb_config=None.
bnb_config = None  # BitsAndBytesConfig(load_in_4bit=True)  # only if the model actually supports it

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    trust_remote_code=True,
    local_files_only=True   # important for custom chat templates/tokenization
)

# Device / dtype: pick ONE of these blocks

# A) CPU, safest
device = "cpu"
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
).to(device)

print("First parameter dtype:", next(model.parameters()).dtype)



messages = [
    {"role": "system", "content": "You are a helpful code review assistant."},
    # {"role": "user", "content": "Review this Python function for clarity and bugs:\n\n"
    #                             "def foo(x):\n    if x==0:\n        return 1\n    return x*foo(x-1)"}
    #{"role": "user", "content": "Can you tell me what is wrong with this code:\n\nconst char* EXE = \"exe\";const int EXELEN = sizeof(EXE);  "}
    {"role": "user", "content": "Can you tell me what is wrong with this C lang code:\n\nint *p = arr;\nchar *q = (char*)p;\nq += 1;\np = (int*)q;\n"}
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
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,   # Playground often uses 0.6–0.8; drop to 0–0.2 for deterministic
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
#exit(0)
with torch.no_grad():
    out = model.generate(**inputs, **gen_kwargs)

#text = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
text = tokenizer.decode(out[0])
print(text)
