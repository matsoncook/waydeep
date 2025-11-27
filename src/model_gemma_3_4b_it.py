

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch,os

import os


class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.path = 'd:/model/'
        self.device = "cpu"
        self.gen_kwargs = dict()
        return

    def load(self):
        os.environ["HF_HOME"] = "D:\\models"
        os.environ["TRANSFORMERS_CACHE"] = "D:\\models"



        MODEL_ID = "google/gemma-3-4b-it"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # or torch.float16 if your GPU doesn’t like bf16
            #device_map="auto",           # puts it on your GPU automatically
        )

        self.model.to("cuda:0")  # <- force everything onto GPU 0

    def input(self, input_str: str) -> str:
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=4096*2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(output)

        return output