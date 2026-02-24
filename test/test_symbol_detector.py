# test/test_symbol_detector.py — Test del Módulo 2 (Symbol Detector)
"""
Verifica que SymbolDetector:
  1. Registra hooks correctamente en las capas LLM
  2. Captura activaciones durante forward pass
  3. Extrae símbolos emergentes via PCA + HDBSCAN
  4. Calcula SCS entre baseline y variantes
"""

import sys
from pathlib import Path

# Asegurar que src/ es importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY
from src.symbol_detector import SymbolDetector


def main():
    print("🔧 Cargando modelo para test de SymbolDetector...")

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

    # ── Test 1: Inicialización y hooks ─────────────────────────────────
    print("\n📊 TEST 1: Crear SymbolDetector y registrar hooks")
    detector = SymbolDetector(model)
    assert len(detector.hooks) > 0, "❌ No se registraron hooks"
    print(f"✅ {len(detector.hooks)} hooks registrados correctamente")

    # ── Test 2: Extraer símbolos ───────────────────────────────────────
    print("\n📊 TEST 2: Extraer símbolos de un prompt")
    prompt = "Analyze this complex image in maximum detail"
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    symbols, latent, variance = detector.extract_symbols(input_ids)

    assert len(symbols) > 0, "❌ No se extrajeron símbolos"
    n_symbols = len(set(symbols[symbols >= 0]))
    print(f"✅ {n_symbols} símbolos emergentes detectados")
    print(f"   Distribución: {symbols}")

    # ── Test 3: Calcular SCS ───────────────────────────────────────────
    print("\n📊 TEST 3: Calcular SCS entre dos extracciones")
    prompt2 = "Describe the main objects visible"
    input_ids2 = processor.tokenizer(prompt2, return_tensors="pt").input_ids.to(model.device)
    symbols2, _, _ = detector.extract_symbols(input_ids2)

    if len(symbols2) > 0:
        scs, metrics = detector.compute_scs(symbols2, symbols)
        print(f"✅ SCS = {scs:.3f}")
        print(f"   Consistency: {metrics['consistency']:.3f}")
        print(f"   Stability:   {metrics['stability']:.3f}")
        print(f"   Cross-Modal: {metrics['cross_modal']:.3f}")
    else:
        print("⚠️ Segunda extracción vacía — skip SCS")

    # ── Cleanup ────────────────────────────────────────────────────────
    detector.remove_hooks()
    print("\n🎉 Todos los tests pasaron. SymbolDetector OK.")


if __name__ == "__main__":
    main()
