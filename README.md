pip install transformers

pip install transformers[torch]

pip install huggingface_hub

pip install transformers accelerate bitsandbytes
pip install optimum
pip install gptqmodel

pip install gptqmodel --no-build-isolation

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

## Caching
C:\Users\markc\.cache\huggingface\hub\models--codellama--CodeLlama-7b-Python-hf. 

:: Use CUDA 12.1 - only if building in c++ etc
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%