from src.model_very_small import Model

model = Model()
model.load()

input_str = input("prompt: ")
output_str = model.input(input_str=input_str)
print(output_str)

input_str = input("prompt1: ")
output_str = model.input(input_str=input_str)
print(output_str)