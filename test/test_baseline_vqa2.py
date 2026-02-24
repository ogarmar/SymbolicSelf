# test/test_baseline_vqa2.py — Test de inferencia LLaVA-v1.6 con offload
"""
Verifica que LLaVA-v1.6-Mistral-7B carga correctamente en 4-bit
con CPU offload y genera respuestas VQA coherentes.

Este test NO usa el pipeline SymbolicSelf — solo valida el modelo base.
"""

import sys
import os
import io
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import requests
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def download_image(url: str) -> Image.Image | None:
    """Descarga y redimensiona imagen para ahorrar VRAM."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.thumbnail((448, 448))
        print(f"✅ Imagen OK: {image.size}")
        return image
    except Exception as e:
        print(f"❌ Error descargando imagen: {e}")
        return None


def main():
    print(f"🔧 Cargando {MODEL_ID} para test baseline...")
    start_load = time.time()

    bnb_config = BitsAndBytesConfig(**QUANTIZATION)
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        max_memory=MAX_MEMORY,
        low_cpu_mem_usage=True,
    )

    print(f"✅ Modelo cargado en {time.time() - start_load:.1f}s")

    # Imagen de test (COCO val2017)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = download_image(image_url)
    if image is None:
        print("❌ No se pudo descargar la imagen de test.")
        return

    questions = [
        "How many cats are there?",
        "Describe the main objects in the image.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*40}")
        print(f"Q{i}: {question}")

        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start_gen = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        response = processor.decode(new_tokens, skip_special_tokens=True).strip()

        vram_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"🤖: {response}")
        print(f"⏱️ {time.time() - start_gen:.2f}s | 💾 VRAM: {vram_gb:.2f}GB")

        torch.cuda.empty_cache()

    print("\n🚀 Test baseline completado.")


if __name__ == "__main__":
    main()
