from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf", local_files_only=True)

model.save_pretrained("./model/code-llama-7b-Python-hf")
tokenizer.save_pretrained("./model/code-llama-7b-Python-hf")

