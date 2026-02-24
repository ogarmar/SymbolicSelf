# download_models.py — Descarga y cachea los modelos necesarios para SymbolicSelf
"""
Descarga LLaVA-v1.6-Mistral-7B en 4-bit y su procesador.
Basta con ejecutarlo una vez; HuggingFace cachea automáticamente.

Uso:
    python download_models.py
"""

import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

print(f"[1/2] Descargando procesador de {MODEL_ID} ...")
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
print("✅ Procesador (tokenizer + image processor) descargado.")

print(f"[2/2] Descargando modelo {MODEL_ID} en 4-bit NF4 ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print(f"✅ Modelo descargado y cacheado por HuggingFace.")
print(f"   Parámetros: {model.num_parameters():,}")
print(f"   VRAM usada: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("🚀 Listo. En próximas ejecuciones se cargará desde caché.")
