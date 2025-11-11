import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# src = r"D:\model\models--openai--gpt-oss-20b\snapshots\6cee5e81ee83917806bbde320786a8fb61efebee"
# dst = r"D:\model\gpt-oss-20b-dequant-bf16"   # new folder for dequantized weights

src = r"/model/gpt-oss-20b"
dst = r"/model/gpt-oss-20b-dequant-bf16"   # new folder for dequantized weights

tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, local_files_only=True)

# IMPORTANT: allow the MXFP4 path here so HF can dequantize the packed MoE tensors.
# Put everything on CPU to avoid CUDA OOM during dequant.
model = AutoModelForCausalLM.from_pretrained(
    src,
    torch_dtype=torch.bfloat16,     # dequant target (source is MXFP4-packed)
    device_map={"": "cpu"},         # force CPU
    trust_remote_code=True,
    local_files_only=True,
)

# Save a *dense*, standard bf16 checkpoint (no MXFP4 blocks/scales)
tok.save_pretrained(dst)
model.save_pretrained(dst, safe_serialization=True)
print("Saved dequantized bf16 to:", dst)