# test/check_gpu.py — Verifica entorno GPU/CUDA
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Memoria total: {props.total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("⚠️ Sin GPU CUDA disponible — se usará CPU.")
