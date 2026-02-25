# test/test_gpu_pipeline.py — Test con GPU cargando el modelo real
"""
Prueba los componentes que necesitan GPU:
  - M2: SymbolDetector (hooks + PCA + HDBSCAN)
  - M1: SelfPolishCore (generación + SCS)
  - Pipeline global: SymbolicSelfPipeline

Requiere:
  - GPU con ≥6GB VRAM
  - Modelo LLaVA descargado
  - Al menos 1 imagen COCO en data/coco/images/val2017/

Uso:
  python test/test_gpu_pipeline.py
  python test/test_gpu_pipeline.py --skip_pipeline   # Solo M2+M1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY
from src.symbol_detector import SymbolDetector
from src.m1_self_polish import SelfPolishCore
from src.m3_self_healing import SelfHealingEngine

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def find_test_image():
    """Busca una imagen COCO para testing."""
    img_dir = Path("data/coco/images/val2017")
    if img_dir.exists():
        images = list(img_dir.glob("*.jpg"))[:1]
        if images:
            return images[0]
    # Fallback: cualquier jpg en el proyecto
    for p in Path("data").rglob("*.jpg"):
        return p
    return None


def load_model():
    """Carga modelo LLaVA 4-bit."""
    print("🔧 Cargando modelo (esto tarda ~30s)...")
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
    return model, processor


# ═══════════════════════════════════════════════════════════════════════
# Test M2: SymbolDetector
# ═══════════════════════════════════════════════════════════════════════

def test_m2_symbol_detector(model, processor, image):
    """Test M2: extracción de símbolos con hooks reales."""
    print("\n🧪 M2: SymbolDetector")

    detector = SymbolDetector(model)
    assert len(detector.hooks) > 0, "No hooks registrados"
    print(f"  ✅ Hooks registrados: {len(detector.hooks)}")

    # Forward pass con imagen
    prompt = "USER: <image>\nWhat do you see? ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.time()
    symbols, raw_acts, variance = detector.extract_symbols(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_sizes=inputs.get("image_sizes"),
    )
    elapsed = (time.time() - t0) * 1000

    n_clusters = len(set(symbols[symbols >= 0]))
    n_noise = int((symbols < 0).sum())
    print(f"  ✅ Symbols: {len(symbols)} tokens, {n_clusters} clusters, {n_noise} noise")
    print(f"  ✅ PCA variance retained: {variance:.3f}")
    print(f"  ✅ Tiempo: {elapsed:.0f}ms")

    # Test SCS
    symbols2 = symbols.copy()
    scs, metrics = detector.compute_scs(symbols, symbols2)
    print(f"  ✅ SCS (self vs self): {scs:.3f}")
    print(f"     Metrics: {metrics}")

    # Verificar vision hook (cross-modal)
    has_vision = "vision_out" in detector.activations
    print(f"  {'✅' if has_vision else '⚠️'} Vision hook: {'activo' if has_vision else 'no disponible (fallback)'}")

    detector.remove_hooks()
    torch.cuda.empty_cache()
    print(f"  ✅ Hooks limpiados")

    return detector, symbols


# ═══════════════════════════════════════════════════════════════════════
# Test M1: SelfPolishCore
# ═══════════════════════════════════════════════════════════════════════

def test_m1_self_polish(model, processor, image):
    """Test M1: generación de variantes y selección por SCS."""
    print("\n🧪 M1: SelfPolishCore")

    detector = SymbolDetector(model)
    polisher = SelfPolishCore(model, processor.tokenizer, detector)

    question = "What color is the sky in this image?"
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")

    t0 = time.time()
    best_response, best_scs, metrics = polisher.run(
        prompt,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        n_variants=2,  # Solo 2 para velocidad
    )
    elapsed = (time.time() - t0) * 1000

    print(f"  ✅ Respuesta: '{best_response[:80]}'")
    print(f"  ✅ SCS: {best_scs:.3f}")
    print(f"  ✅ Metrics: {metrics}")
    print(f"  ✅ Tiempo: {elapsed:.0f}ms")

    detector.remove_hooks()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Test M3: SelfHealing (con símbolos reales)
# ═══════════════════════════════════════════════════════════════════════

def test_m3_with_real_symbols(model, processor, image):
    """Test M3: diagnóstico con símbolos reales del modelo."""
    print("\n🧪 M3: SelfHealing (con símbolos reales)")

    detector = SymbolDetector(model)
    healer = SelfHealingEngine(detector)

    # Extraer símbolos de 2 prompts diferentes
    prompts = [
        "USER: <image>\nWhat is in this image? ASSISTANT:",
        "USER: <image>\nDescribe the colors. ASSISTANT:",
    ]

    all_symbols = []
    for prompt in prompts:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        symbols, _, _ = detector.extract_symbols(
            inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_sizes=inputs.get("image_sizes"),
        )
        all_symbols.append(symbols)
        torch.cuda.empty_cache()

    # Establecer baseline con primer prompt
    healer.establish_baseline(all_symbols[0])
    print(f"  ✅ Baseline: {len(all_symbols[0])} symbols")

    # Diagnosticar con segundo prompt
    diag = healer.diagnose(all_symbols[1])
    print(f"  ✅ Diagnóstico: status={diag.status.value}")
    print(f"     stability={diag.stability:.3f}, entropy_delta={diag.entropy_delta:.3f}")
    print(f"     action='{diag.healing_action[:60]}'")

    detector.remove_hooks()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Test Pipeline Global
# ═══════════════════════════════════════════════════════════════════════

def test_pipeline_global(image):
    """Test completo: SymbolicSelfPipeline end-to-end."""
    print("\n🧪 Pipeline Global: SymbolicSelfPipeline")

    from src.symbolic_self import SymbolicSelfPipeline

    t0 = time.time()
    pipeline = SymbolicSelfPipeline()
    load_time = time.time() - t0
    print(f"  ✅ Pipeline inicializado en {load_time:.1f}s")

    # Primera llamada (establece baseline)
    t0 = time.time()
    result = pipeline.process("What do you see in this image?", image, n_variants=2)
    elapsed = (time.time() - t0) * 1000

    print(f"  ✅ Resultado 1:")
    print(f"     response: '{result.response[:80]}'")
    print(f"     scs: {result.scs:.3f}")
    print(f"     diagnosis: {result.diagnosis}")
    print(f"     latency: {result.latency_ms:.0f}ms")

    # Segunda llamada (diagnostica contra baseline)
    result2 = pipeline.process("What colors are present?", image, n_variants=2)
    print(f"  ✅ Resultado 2:")
    print(f"     response: '{result2.response[:80]}'")
    print(f"     diagnosis: {result2.diagnosis}")
    print(f"     healing: '{result2.healing_action[:60]}'")

    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_pipeline", action="store_true", help="Saltar test del pipeline global")
    args = parser.parse_args()

    # Buscar imagen
    img_path = find_test_image()
    if img_path is None:
        print("❌ No se encontró imagen de test. Necesitas al menos 1 .jpg en data/coco/images/val2017/")
        sys.exit(1)

    print(f"📷 Imagen de test: {img_path}")
    image = Image.open(img_path).convert("RGB")

    # Cargar modelo (compartido entre tests)
    model, processor = load_model()

    print(f"\n{'='*70}")
    print(f"  TEST SUITE CON GPU — SymbolicSelf")
    print(f"{'='*70}")

    passed = 0
    failed = 0

    # Test M2
    try:
        test_m2_symbol_detector(model, processor, image)
        passed += 1
    except Exception as e:
        print(f"  ❌ M2 FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Test M1
    try:
        test_m1_self_polish(model, processor, image)
        passed += 1
    except Exception as e:
        print(f"  ❌ M1 FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Test M3
    try:
        test_m3_with_real_symbols(model, processor, image)
        passed += 1
    except Exception as e:
        print(f"  ❌ M3 FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Liberar modelo antes del pipeline (que carga uno nuevo)
    del model
    torch.cuda.empty_cache()

    # Test Pipeline Global
    if not args.skip_pipeline:
        try:
            test_pipeline_global(image)
            passed += 1
        except Exception as e:
            print(f"  ❌ Pipeline FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    else:
        print("\n⏭️ Pipeline global: SKIPPED")

    print(f"\n{'='*70}")
    print(f"  RESULTADO: {passed} passed, {failed} failed")
    print(f"{'='*70}")

    torch.cuda.empty_cache()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
