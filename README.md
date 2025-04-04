pip install transformers

pip install transformers[torch]

pip install huggingface_hub

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


model.safetensors.index.json: 100%|███████████████████████████████████████████████| 25.1k/25.1k [00:00<00:00, 25.2MB/s]
C:\dev\llama\.venv\Lib\site-packages\huggingface_hub\file_download.py:142: 
UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store 
duplicated files but your machine does not support them in 
C:\Users\markc\.cache\huggingface\hub\models--codellama--CodeLlama-7b-Python-hf. 
Caching files will still work but in a degraded version that might require more space on your disk. 
This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. 
For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. 
In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development