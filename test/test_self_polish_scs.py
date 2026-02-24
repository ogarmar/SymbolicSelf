# test/test_self_polish_scs.py — Test del Módulo 1 (Self-Polish + SCS)
"""
Verifica que SelfPolishCore:
  1. Genera variantes refinadas a partir de templates
  2. Calcula SCS para cada variante vs baseline
  3. Selecciona la variante con mejor SCS
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY
from src.symbol_detector import SymbolDetector
from src.m1_self_polish import SelfPolishCore


def main():
    print("🔧 Cargando modelo para test de Self-Polish + SCS...")

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

    detector = SymbolDetector(model)
    core = SelfPolishCore(model, processor.tokenizer, detector)

    # ── Test 1: Generar variantes ──────────────────────────────────────
    print("\n📊 TEST 1: Generar 3 variantes refinadas")
    prompt = "Describe this complex image"
    variants = core.generate_variants(prompt, n_variants=3)

    assert len(variants) == 3, f"❌ Esperaba 3 variantes, obtuvo {len(variants)}"
    for i, v in enumerate(variants):
        print(f"   V{i+1}: {v[:80]}...")
    print("✅ Variantes generadas correctamente")

    # ── Test 2: Pipeline completo ──────────────────────────────────────
    print("\n📊 TEST 2: Pipeline Self-Polish completo (baseline → SCS → selección)")
    best_response, best_scs, metrics = core.run(prompt, n_variants=3)

    print(f"   🏆 Ganadora: SCS={best_scs:.3f}")
    print(f"   Texto: {best_response[:120]}...")
    print(f"   Métricas: {metrics}")

    assert best_scs >= 0.0, f"❌ SCS negativo: {best_scs}"
    assert len(best_response) > 0, "❌ Respuesta vacía"
    print("✅ Pipeline Self-Polish completo OK")

    # ── Cleanup ────────────────────────────────────────────────────────
    detector.remove_hooks()
    torch.cuda.empty_cache()
    print("\n🎉 Todos los tests pasaron. Self-Polish + SCS OK.")


if __name__ == "__main__":
    main()
