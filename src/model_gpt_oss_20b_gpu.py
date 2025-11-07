import torch, os
from transformers import AutoTokenizer,AutoConfig, AutoModelForCausalLM

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
        model_path = r"D:\model\gpt-oss-20b-dequant-bf16"
        # If you’re on GPU and the repo ships 4-bit weights, keep this; otherwise set bnb_config=None.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

        # 1) Load/clean config to ensure no MXFP4 is requested
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        if hasattr(cfg, "quantization_config"):
            cfg.quantization_config = None
            if hasattr(cfg, "__dict__"):
                cfg.__dict__.pop("quantization_config", None)

        # bnb_cfg = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,   # T4 uses fp16 compute
        #     bnb_4bit_quant_type="nf4",              # good default
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,

            trust_remote_code=True,
            local_files_only=True  # important for custom chat templates/tokenization
        )
        max_memory = {0: "12GiB", 1: "14GiB", 2: "14GiB", "cpu": "48GiB"}  # optional CPU offload

        # # strip any quantization hints from the config object
        # if hasattr(tokenizer, "quantization_config"):
        #     tokenizer.quantization_config = None
        #     # extra guard for raw dicts:
        #     if hasattr(tokenizer, "__dict__"):
        #         tokenizer.__dict__.pop("quantization_config", None)

        # Device / dtype: pick ONE of these blocks

        # # A) CPU, safest
        # device = "cpu"
        # dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            dtype=torch.float16,  # T4 uses FP16 (not BF16)
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            # variant="bf16",              # or "fp16" if that exists
            # ignore_patterns=["*mxfp4*", "*MXFP4*", "*mx*"],
            offload_folder=r"D:\offload_tmp",  # helps peak mem during load
            attn_implementation="eager",
        )  # .to(device)

        print("First parameter dtype:", next(self.model.parameters()).dtype)

    def input(self,input_str: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful code review assistant."},
            # {"role": "user", "content": "Review this Python function for clarity and bugs:\n\n"
            #                             "def foo(x):\n    if x==0:\n        return 1\n    return x*foo(x-1)"}
            {"role": "user",
             "content": "Can you tell me what is wrong with this code:\n\nconst char* EXE = \"exe\";const int EXELEN = sizeof(EXE);  "}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # appends the model’s expected assistant prefix
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")  # .to(device)

        # Make sure eos/pad are sane
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,  # Playground often uses 0.6–0.8; drop to 0–0.2 for deterministic
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # exit(0)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(text)
        return text
