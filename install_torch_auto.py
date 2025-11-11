import os
import subprocess
import sys

TORCH_VERSION = "2.5.1"
VISION_VERSION = "0.20.1"
AUDIO_VERSION = "2.5.1"
PIP = "pip"

def has_nvidia_gpu():
    try:
        # Try running nvidia-smi to detect a CUDA-capable GPU
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def install_torch(cuda_available):
    if cuda_available:
        # CUDA 12.1 build (change cu121 to another variant if desired)
        url = "https://download.pytorch.org/whl/cu121"
        print("✅ NVIDIA GPU detected — installing CUDA 12.1 build of PyTorch")
    else:
        # CPU-only build
        url = "https://download.pytorch.org/whl/cpu"
        print("⚙️  No GPU detected — installing CPU-only build of PyTorch")

    cmd = [
        sys.executable, "-m", PIP, "install", "--upgrade",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={VISION_VERSION}",
        f"torchaudio=={AUDIO_VERSION}",
        "--index-url", url
    ]

    print("📦 Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    gpu = has_nvidia_gpu()
    install_torch(gpu)
    print("✅ Installation complete!")