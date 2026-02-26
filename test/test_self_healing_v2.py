# test/test_self_healing_v2.py — Test del Módulo 3 (Self-Healing) con imagen
"""
Verifica que SelfHealingEngine con entrada MULTIMODAL:
  1. Establece un baseline simbólico rico (con clusters reales)
  2. Detecta degradación adversarial (símbolos perturbados)
  3. Detecta concept drift (símbolos desplazados)
  4. Confirma estado saludable (mismos símbolos = stable)
  5. Entropía calculada correctamente
"""

import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from src.model_loader import load_model_sync
from src.symbol_detector import SymbolDetector
from src.m3_self_healing import SelfHealingEngine, DegradationType
from test.utils import download_image


IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

def main():
    print("🔧 Cargando modelo para test visual de Self-Healing...")

    model, processor = load_model_sync()

    image = download_image(IMAGE_URL)
    print(f"✅ Imagen descargada: {image.size}")

    detector = SymbolDetector(model)
    healer = SelfHealingEngine(detector)

    # ── Test 1: Establecer baseline con imagen ─────────────────────────
    print("\n📊 TEST 1: Establecer baseline simbólico con imagen")
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    baseline_symbols, _, _ = detector.extract_symbols(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_sizes=inputs.get("image_sizes"),
    )

    n_baseline = len(set(baseline_symbols[baseline_symbols >= 0]))
    print(f"   Baseline: {len(baseline_symbols)} tokens, {n_baseline} clusters")
    print(f"   Distribución: {baseline_symbols[:30]}...")

    healer.establish_baseline(baseline_symbols)
    print(f"✅ Baseline establecido ({n_baseline} símbolos)")

    # ── Test 2: Estado saludable (mismos símbolos) ─────────────────────
    print("\n📊 TEST 2: Verificar estado saludable (misma extracción)")
    diagnosis = healer.diagnose(baseline_symbols)
    print(f"   {diagnosis}")
    print(f"   Stability: {diagnosis.stability:.3f}")

    # Con mismos símbolos, stability debe ser 1.0 → HEALTHY
    assert diagnosis.status == DegradationType.HEALTHY, \
        f"❌ Esperaba HEALTHY con símbolos idénticos, obtuvo {diagnosis.status}"
    print("✅ Estado saludable confirmado")

    # ── Test 3: Extracción con pregunta diferente ──────────────────────
    print("\n📊 TEST 3: Extracción con pregunta diferente (misma imagen)")
    prompt2 = "USER: <image>\nWhat animals are in this image? ASSISTANT:"
    inputs2 = processor(text=prompt2, images=image, return_tensors="pt")
    inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

    symbols2, _, _ = detector.extract_symbols(
        inputs2["input_ids"],
        pixel_values=inputs2.get("pixel_values"),
        image_sizes=inputs2.get("image_sizes"),
    )

    n_sym2 = len(set(symbols2[symbols2 >= 0]))
    print(f"   Segunda extracción: {n_sym2} clusters")

    diagnosis2 = healer.diagnose(symbols2)
    print(f"   {diagnosis2}")
    print(f"   Status: {diagnosis2.status.value}")

    # ── Test 4: Simular ADVERSARIAL ────────────────────────────────────
    print("\n📊 TEST 4: Simular Ataque Adversarial")
    adv_symbols = healer.simulate_degradation("adversarial", severity=1.2)
    diagnosis_adv = healer.diagnose(adv_symbols)
    print(f"   {diagnosis_adv}")
    assert diagnosis_adv.stability < 0.6, \
        f"❌ Stability demasiado alta para adversarial: {diagnosis_adv.stability}"
    print("✅ Adversarial detectado correctamente")

    # ── Test 5: Simular DRIFT ──────────────────────────────────────────
    print("\n📊 TEST 5: Simular Concept Drift")
    drift_symbols = healer.simulate_degradation("drift", severity=0.7)
    diagnosis_drift = healer.diagnose(drift_symbols)
    print(f"   {diagnosis_drift}")
    assert diagnosis_drift.status in (DegradationType.DRIFT, DegradationType.ADVERSARIAL), \
        f"❌ Esperaba drift/adversarial, obtuvo {diagnosis_drift.status}"
    print("✅ Drift detectado correctamente")

    # ── Test 6: Entropía correcta ──────────────────────────────────────
    print("\n📊 TEST 6: Verificar cálculo de entropía")
    # Distribución uniforme de 4 clusters → entropía = 2.0
    uniform = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    e_uniform = SelfHealingEngine._compute_entropy(uniform)
    assert abs(e_uniform - 2.0) < 0.01, f"❌ Entropía incorrecta: {e_uniform}"
    print(f"✅ Entropía uniforme = {e_uniform:.3f} (esperada 2.000)")

    # Distribución degenerada → entropía 0
    single = np.array([0, 0, 0, 0])
    e_single = SelfHealingEngine._compute_entropy(single)
    assert e_single < 0.01, f"❌ Entropía debería ser ~0: {e_single}"
    print(f"✅ Entropía single = {e_single:.3f} (esperada 0.000)")

    # ── Cleanup ────────────────────────────────────────────────────────
    detector.remove_hooks()
    torch.cuda.empty_cache()
    print(f"\n🎉 Todos los tests visuales pasaron. Self-Healing V2 OK.")
    print(f"   Historial: {len(healer.history)} diagnósticos registrados")


if __name__ == "__main__":
    main()
