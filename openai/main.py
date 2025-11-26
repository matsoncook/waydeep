from src.model import Model

import torch
print("torch.cuda.is_available(): ",torch.cuda.is_available())  # Should be True
print("torch.cuda.device_count() ",torch.cuda.device_count())  # Should be 3

model = Model()
model.load()

input_str = input("prompt: ")
output_str = model.input(input_str=input_str)
print(output_str)

input_str = input("prompt1: ")
output_str = model.input(input_str=input_str)
print(output_str)