
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.path = 'd:/model/'
        self.device = "cpu"
        self.gen_kwargs = dict()
        return

    def load(self):
#model_id = "openai/gpt-oss-20b"  # your repo

        model_path = r"D:\model\models--openai--gpt-oss-20b\snapshots\6cee5e81ee83917806bbde320786a8fb61efebee"
        # If you’re on GPU and the repo ships 4-bit weights, keep this; otherwise set bnb_config=None.
        bnb_config = None  # BitsAndBytesConfig(load_in_4bit=True)  # only if the model actually supports it

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True   # important for custom chat templates/tokenization
        )

        # Device / dtype: pick ONE of these blocks

        # A) CPU, safest

        dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        ).to(self.device)




        # Make sure eos/pad are sane
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.gen_kwargs = dict(
            max_new_tokens=8192, # how many new tokens to generate
            do_sample=True,
            temperature=0.6,  # Playground often uses 0.6–0.8; drop to 0–0.2 for deterministic
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        print("First parameter dtype:", next(self.model.parameters()).dtype)
        print("Context length: ", self.model.config.max_position_embeddings)

    def input(self,input_str: str) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful code review assistant."},
            # {"role": "user", "content": "Review this Python function for clarity and bugs:\n\n"
            #                             "def foo(x):\n    if x==0:\n        return 1\n    return x*foo(x-1)"}
            #{"role": "user", "content": "Can you tell me what is wrong with this code:\n\nconst char* EXE = \"exe\";const int EXELEN = sizeof(EXE);  "}
            {"role": "user", "content": input_str}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True      # appends the model’s expected assistant prefix
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)


        #exit(0)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self.gen_kwargs)

        text = self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(text)

        return text
