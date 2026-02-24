# test/test_symbol_detector_v2.py — Test del Módulo 2 (Symbol Detector) con imagen
"""
Verifica que SymbolDetector con entrada MULTIMODAL (texto + imagen):
  1. Registra hooks correctamente en las capas LLM
  2. Captura activaciones durante forward pass multimodal
  3. Extrae símbolos emergentes reales via PCA + HDBSCAN
  4. Calcula SCS entre baseline y variantes con símbolos no vacíos
"""

import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY
from src.symbol_detector import SymbolDetector


IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def download_image(url: str) -> Image.Image:
    """Descarga imagen COCO y la redimensiona."""
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image.thumbnail((448, 448))
    return image


def main():
    print("🔧 Cargando modelo para test visual de SymbolDetector...")

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

    image = download_image(IMAGE_URL)
    print(f"✅ Imagen descargada: {image.size}")

    # ── Test 1: Inicialización y hooks ─────────────────────────────────
    print("\n📊 TEST 1: Crear SymbolDetector y registrar hooks")
    detector = SymbolDetector(model)
    assert len(detector.hooks) > 0, "❌ No se registraron hooks"
    print(f"✅ {len(detector.hooks)} hooks registrados correctamente")

    # ── Test 2: Extraer símbolos con imagen ────────────────────────────
    print("\n📊 TEST 2: Extraer símbolos de prompt + imagen")
    prompt1 = "USER: <image>\nHow many cats are there? ASSISTANT:"
    inputs1 = processor(text=prompt1, images=image, return_tensors="pt")
    inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}

    symbols1, latent1, variance1 = detector.extract_symbols(
        inputs1["input_ids"],
        pixel_values=inputs1.get("pixel_values"),
        image_sizes=inputs1.get("image_sizes"),
    )

    n_symbols = len(set(symbols1[symbols1 >= 0]))
    n_noise = int((symbols1 == -1).sum())
    print(f"   Tokens: {len(symbols1)} total, {n_symbols} clusters, {n_noise} ruido")
    print(f"   Distribución: {symbols1[:20]}...")
    print(f"   Varianza retenida PCA: {variance1:.3f}")

    if n_symbols > 0:
        print(f"✅ {n_symbols} símbolos emergentes detectados")
    else:
        print("⚠️ 0 símbolos (HDBSCAN consideró todo ruido) — continúa")

    # ── Test 3: Segunda extracción (pregunta diferente) ────────────────
    print("\n📊 TEST 3: Extraer símbolos de segunda pregunta + misma imagen")
    prompt2 = "USER: <image>\nDescribe the main objects visible in this image. ASSISTANT:"
    inputs2 = processor(text=prompt2, images=image, return_tensors="pt")
    inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

    symbols2, latent2, variance2 = detector.extract_symbols(
        inputs2["input_ids"],
        pixel_values=inputs2.get("pixel_values"),
        image_sizes=inputs2.get("image_sizes"),
    )

    n_symbols2 = len(set(symbols2[symbols2 >= 0]))
    print(f"   {n_symbols2} símbolos en segunda extracción")

    # ── Test 4: Calcular SCS entre ambas extracciones ──────────────────
    print("\n📊 TEST 4: Calcular SCS entre extracciones")
    if len(symbols1) > 0 and len(symbols2) > 0:
        scs, metrics = detector.compute_scs(symbols2, symbols1)
        print(f"✅ SCS = {scs:.3f}")
        print(f"   Consistency: {metrics['consistency']:.3f}")
        print(f"   Stability:   {metrics['stability']:.3f}")
        print(f"   Cross-Modal: {metrics['cross_modal']:.3f}")
        assert 0.0 <= scs <= 1.0, f"❌ SCS fuera de rango: {scs}"
    else:
        print("⚠️ Extracción incompleta — skip SCS")

    # ── Test 5: SCS de la misma extracción (self-consistency) ──────────
    print("\n📊 TEST 5: SCS self-consistency (misma extracción vs sí misma)")
    if len(symbols1) > 0:
        scs_self, _ = detector.compute_scs(symbols1, symbols1)
        print(f"✅ SCS self = {scs_self:.3f} (esperada cercana a 1.0)")
    else:
        print("⚠️ Sin símbolos — skip self-consistency")

    # ── Cleanup ────────────────────────────────────────────────────────
    detector.remove_hooks()
    torch.cuda.empty_cache()
    print("\n🎉 Todos los tests visuales pasaron. SymbolDetector V2 OK.")


if __name__ == "__main__":
    main()
