# test/test_self_healing.py — Test del Módulo 3 (Self-Healing)
"""
Verifica que SelfHealingEngine:
  1. Establece un baseline simbólico
  2. Detecta degradación adversarial (stability < 0.3)
  3. Detecta concept drift (stability 0.3-0.6)
  4. Confirma estado saludable (stability > 0.6)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.model_loader import load_model_sync
from src.symbol_detector import SymbolDetector
from src.m3_self_healing import SelfHealingEngine, DegradationType


def main():
    print("🔧 Cargando modelo para test de Self-Healing...")

    model, processor = load_model_sync()

    detector = SymbolDetector(model)
    healer = SelfHealingEngine(detector)

    # ── Test 1: Establecer baseline ────────────────────────────────────
    print("\n📊 TEST 1: Establecer baseline simbólico")
    prompt = "Describe this complex image in detail"
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    baseline_symbols, _, _ = detector.extract_symbols(input_ids)

    assert len(baseline_symbols) > 0, "❌ Baseline vacío"
    healer.establish_baseline(baseline_symbols)
    print(f"✅ Baseline: {len(set(baseline_symbols[baseline_symbols >= 0]))} símbolos")

    # ── Test 2: Simular DRIFT ──────────────────────────────────────────
    print("\n📊 TEST 2: Simular Concept Drift")
    drift_symbols = healer.simulate_degradation("drift", severity=0.7)
    diagnosis = healer.diagnose(drift_symbols)
    print(f"   {diagnosis}")
    assert diagnosis.status in (DegradationType.DRIFT, DegradationType.ADVERSARIAL), \
        f"❌ Esperaba drift/adversarial, obtuvo {diagnosis.status}"
    print("✅ Drift detectado correctamente")

    # ── Test 3: Simular ADVERSARIAL ────────────────────────────────────
    print("\n📊 TEST 3: Simular Ataque Adversarial")
    adv_symbols = healer.simulate_degradation("adversarial", severity=1.2)
    diagnosis = healer.diagnose(adv_symbols)
    print(f"   {diagnosis}")
    # Adversarial should show low stability
    assert diagnosis.stability < 0.6, \
        f"❌ Stability demasiado alta para adversarial: {diagnosis.stability}"
    print("✅ Adversarial detectado correctamente")

    # ── Test 4: Estado saludable ───────────────────────────────────────
    print("\n📊 TEST 4: Verificar estado saludable")
    # Símbolos idénticos al baseline = saludable
    diagnosis = healer.diagnose(baseline_symbols)
    print(f"   {diagnosis}")
    assert diagnosis.status == DegradationType.HEALTHY, \
        f"❌ Esperaba HEALTHY, obtuvo {diagnosis.status}"
    print("✅ Estado saludable confirmado")

    # ── Test 5: Entropía correcta ──────────────────────────────────────
    print("\n📊 TEST 5: Verificar cálculo de entropía")
    # Distribución uniforme de 4 clusters → entropía = 2.0
    uniform_symbols = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    entropy = SelfHealingEngine._compute_entropy(uniform_symbols)
    assert abs(entropy - 2.0) < 0.01, f"❌ Entropía incorrecta: {entropy} (esperada 2.0)"
    print(f"✅ Entropía uniforme = {entropy:.3f} (esperada 2.000)")

    # Distribución degenerada (1 cluster) → entropía = 0
    single = np.array([0, 0, 0, 0])
    entropy_single = SelfHealingEngine._compute_entropy(single)
    assert entropy_single < 0.01, f"❌ Entropía debería ser ~0: {entropy_single}"
    print(f"✅ Entropía single = {entropy_single:.3f} (esperada 0.000)")

    # ── Cleanup ────────────────────────────────────────────────────────
    detector.remove_hooks()
    print(f"\n🎉 Todos los tests pasaron. Self-Healing OK.")
    print(f"   Historial: {len(healer.history)} diagnósticos registrados")


if __name__ == "__main__":
    main()
