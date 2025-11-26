from transformers import AutoModelForCausalLM, AutoTokenizer
import torch,os

import os
os.environ["HF_HOME"] = "D:\\models"
os.environ["TRANSFORMERS_CACHE"] = "D:\\models"



MODEL_ID = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # or torch.float16 if your GPU doesn’t like bf16
    device_map="auto",           # puts it on your GPU automatically
)

prompt = "Explain what a TCP socket is in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

gen_kwargs = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))