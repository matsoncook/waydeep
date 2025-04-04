from transformers import AutoTokenizer, AutoModelForCausalLM

class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        return

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
        self.model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")





        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print("device",next(self.model.parameters()).device)

    def input(self,input_str: str) -> str:
        inputs = self.tokenizer(input_str, return_tensors="pt")



        outputs = self.model.generate(

            inputs["input_ids"],
            max_new_tokens=100,  # number of tokens to generate
            temperature=0.7,  # control randomness (lower = less random)
            do_sample=True,  # set True for sampling-based generation
            top_p=0.95,  # nucleus sampling, for diverse yet coherent output
        )

        # Decode and print the result
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
