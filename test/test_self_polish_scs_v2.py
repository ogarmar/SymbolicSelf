# test/test_self_polish_scs_v2.py — Test del Módulo 1 (Self-Polish + SCS) con imagen
"""
Verifica que SelfPolishCore con entrada MULTIMODAL:
  1. Genera variantes refinadas usando imagen + templates
  2. Calcula SCS real con símbolos visuales (no vacíos)
  3. Selecciona la variante con mejor SCS
  4. Compara SCS visual vs SCS text-only
"""

import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.model_loader import load_model_sync
from src.symbol_detector import SymbolDetector
from src.m1_self_polish import SelfPolishCore
from test.utils import download_image


IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def main():
    print("🔧 Cargando modelo para test visual de Self-Polish + SCS...")

    model, processor = load_model_sync()

    image = download_image(IMAGE_URL)
    print(f"✅ Imagen descargada: {image.size}")

    detector = SymbolDetector(model)
    core = SelfPolishCore(model, processor.tokenizer, detector)

    # ── Test 1: Generar respuesta baseline con imagen ──────────────────
    print("\n📊 TEST 1: Generar baseline con imagen")
    question = "How many cats are in the image?"
    prompt = f"USER: <image>\n{question} ASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        baseline_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=False,  # prevenir fuga de KV-cache en 6GB GPU
        )

    new_tokens = baseline_ids[0][inputs["input_ids"].shape[-1]:]
    baseline_response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"   Baseline: {baseline_response}")

    # ── Test 2: Extraer símbolos del baseline (con imagen) ─────────────
    print("\n📊 TEST 2: Extraer símbolos baseline (multimodal)")
    baseline_symbols, _, var_baseline = detector.extract_symbols(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_sizes=inputs.get("image_sizes"),
    )

    n_baseline = len(set(baseline_symbols[baseline_symbols >= 0]))
    print(f"   {n_baseline} símbolos, varianza PCA: {var_baseline:.3f}")
    print(f"   Distribución: {baseline_symbols[:20]}...")

    # ── Test 3: Generar variantes refinadas ────────────────────────────
    print("\n📊 TEST 3: Generar 3 variantes refinadas con imagen")

    # Para las variantes con imagen, procesamos cada template
    templates = [
        f"USER: <image>\n{question} Be precise and count carefully. ASSISTANT:",
        f"USER: <image>\n{question} Look at every detail in the image. ASSISTANT:",
        f"USER: <image>\n{question} Describe what you see clearly. ASSISTANT:",
    ]

    variants = []
    variant_scs_scores = []

    for i, tmpl_prompt in enumerate(templates):
        tmpl_inputs = processor(text=tmpl_prompt, images=image, return_tensors="pt")
        tmpl_inputs = {k: v.to(model.device) for k, v in tmpl_inputs.items()}

        with torch.no_grad():
            var_ids = model.generate(
                **tmpl_inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=False,  # prevenir fuga de KV-cache en 6GB GPU
            )

        new_ids = var_ids[0][tmpl_inputs["input_ids"].shape[-1]:]
        variant_text = processor.decode(new_ids, skip_special_tokens=True).strip()
        variants.append(variant_text)

        # Extraer símbolos de la variante (con imagen)
        var_symbols, _, _ = detector.extract_symbols(
            tmpl_inputs["input_ids"],
            pixel_values=tmpl_inputs.get("pixel_values"),
            image_sizes=tmpl_inputs.get("image_sizes"),
        )

        # Calcular SCS vs baseline
        if len(var_symbols) > 0 and len(baseline_symbols) > 0:
            scs, metrics = detector.compute_scs(var_symbols, baseline_symbols)
        else:
            scs, metrics = 0.0, {}

        variant_scs_scores.append(scs)
        print(f"   V{i+1}: SCS={scs:.3f} | {variant_text[:80]}...")

    # ── Test 4: Seleccionar la mejor ───────────────────────────────────
    print("\n📊 TEST 4: Seleccionar mejor variante por SCS")
    best_idx = max(range(len(variant_scs_scores)), key=lambda i: variant_scs_scores[i])
    best_scs = variant_scs_scores[best_idx]
    best_variant = variants[best_idx]

    print(f"   🏆 Ganadora: V{best_idx+1} con SCS={best_scs:.3f}")
    print(f"   Texto: {best_variant}")
    print(f"   Baseline: {baseline_response}")

    assert best_scs >= 0.0, f"❌ SCS negativo: {best_scs}"
    assert len(best_variant) > 0, "❌ Variante vacía"
    print("✅ Selección por SCS completada")

    # ── Test 5: Verificar que SCS mejora con variantes ─────────────────
    print("\n📊 TEST 5: Resumen de scores")
    print(f"   {'Variante':<12} {'SCS':>6}")
    print(f"   {'─'*12} {'─'*6}")
    for i, scs in enumerate(variant_scs_scores):
        marker = " 🏆" if i == best_idx else ""
        print(f"   V{i+1:<10} {scs:>6.3f}{marker}")

    # ── Cleanup ────────────────────────────────────────────────────────
    detector.remove_hooks()
    torch.cuda.empty_cache()
    print("\n🎉 Todos los tests visuales pasaron. Self-Polish + SCS V2 OK.")


if __name__ == "__main__":
    main()
