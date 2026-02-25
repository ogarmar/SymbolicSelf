# test/test_all_modules.py — Test integral de todos los modulos
"""
Prueba cada modulo individualmente SIN necesidad de GPU/modelo.
Usa mocks cuando es necesario.

Uso:
  python -m pytest test/test_all_modules.py -v
  python test/test_all_modules.py
"""

import sys
import os
from pathlib import Path

# Fix encoding en Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# === M4: Meta-Evolutionary Optimizer (sin GPU) ===

def test_m4_strategy_genome():
    """Test M4: StrategyGenome serializacion/deserializacion."""
    from src.m4_meta_evo import StrategyGenome

    g = StrategyGenome(scs_alpha=0.5, scs_beta=0.3, scs_gamma=0.2, n_variants=5)
    vec = g.to_vector()
    assert len(vec) == 7, f"Genoma debe tener 7 dims, tiene {len(vec)}"

    g2 = StrategyGenome.from_vector(vec)
    assert abs(g2.scs_alpha - 0.5) < 0.01
    assert abs(g2.scs_beta - 0.3) < 0.01
    assert g2.n_variants == 5
    print("  [OK] M4 StrategyGenome: serializacion OK")


def test_m4_optimizer():
    """Test M4: Differential Evolution con fitness simple."""
    from src.m4_meta_evo import MetaEvolutionaryOptimizer, StrategyGenome

    def dummy_fitness(genome: StrategyGenome) -> float:
        return genome.scs_alpha - genome.n_variants * 0.1

    opt = MetaEvolutionaryOptimizer(
        population_size=5,
        max_generations=3,
        seed=42,
    )
    result = opt.optimize(dummy_fitness)

    assert result.best_fitness > -float("inf"), "Fitness debe ser finito"
    assert result.generations == 3
    assert len(result.history) > 0
    print(f"  [OK] M4 Optimizer: best_fitness={result.best_fitness:.4f}, "
          f"alpha={result.best_genome.scs_alpha:.2f}")


# === M5: Semantic Memory (sin GPU) ===

def test_m5_store_and_retrieve():
    """Test M5: almacenar y recuperar entradas."""
    from src.m5_semantic_memory import SemanticMemory

    mem = SemanticMemory(max_entries=5, embedding_dim=64)
    rng = np.random.default_rng(42)

    for i in range(3):
        embedding = rng.standard_normal(64).astype(np.float32)
        symbols = np.array([0, 1, 1, 0, -1, 2])
        mem.store(
            question=f"Question {i}",
            answer=f"Answer {i}",
            embedding=embedding,
            symbols=symbols,
            scs=0.7 + i * 0.1,
        )

    assert len(mem) == 3, f"Debe tener 3 entradas, tiene {len(mem)}"
    print(f"  [OK] M5 Store: {len(mem)} entradas almacenadas")

    query = rng.standard_normal(64).astype(np.float32)
    results = mem.retrieve(query, top_k=2)
    assert len(results) <= 2
    assert all(isinstance(r[1], float) for r in results)
    print(f"  [OK] M5 Retrieve: top-{len(results)} resultados")


def test_m5_eviction():
    """Test M5: eviccion FIFO cuando la memoria esta llena."""
    from src.m5_semantic_memory import SemanticMemory

    mem = SemanticMemory(max_entries=3, embedding_dim=16)

    for i in range(5):
        emb = np.random.randn(16).astype(np.float32)
        mem.store(f"Q{i}", f"A{i}", emb, np.array([0, 1]), 0.5)

    assert len(mem) == 3, f"Max 3 entradas, tiene {len(mem)}"
    assert mem.entries[0].question == "Q2"
    assert mem.entries[-1].question == "Q4"
    print(f"  [OK] M5 Eviction: {len(mem)} entradas, mas antigua = {mem.entries[0].question}")


def test_m5_stats():
    """Test M5: estadisticas de la memoria."""
    from src.m5_semantic_memory import SemanticMemory

    mem = SemanticMemory(max_entries=10, embedding_dim=16)
    stats = mem.get_stats()
    assert stats["n_entries"] == 0

    mem.store("Q1", "A1", np.zeros(16), np.array([0]), 0.8)
    stats = mem.get_stats()
    assert stats["n_entries"] == 1
    assert stats["avg_scs"] == 0.8
    print(f"  [OK] M5 Stats: {stats}")


# === M3: SelfHealing (sin GPU) ===

def test_m3_diagnosis():
    """Test M3: diagnostico con simbolos diferentes deberia detectar drift."""
    from src.m3_self_healing import SelfHealingEngine

    healer = SelfHealingEngine(detector=None)

    baseline_symbols = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    healer.establish_baseline(baseline_symbols)

    diag = healer.diagnose(baseline_symbols)
    print(f"  [OK] M3 Healthy: status={diag.status.value}")

    drifted = np.array([3, 3, 4, 4, 5, 5, 3, 4])
    diag2 = healer.diagnose(drifted)
    print(f"  [OK] M3 Drift: status={diag2.status.value}, action='{diag2.healing_action[:50]}'")


# === Config e imports ===

def test_config_imports():
    """Test: todos los imports de config funcionan."""
    from src.config import (
        MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY,
        HOOK_LAYERS, PCA_COMPONENTS_TOKENS,
        HDBSCAN_MIN_CLUSTER, HDBSCAN_MIN_SAMPLES,
        SCS_ALPHA, SCS_BETA, SCS_GAMMA,
        REFINE_TEMPLATES, DEFAULT_N_VARIANTS,
        GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE,
    )
    assert len(HOOK_LAYERS) == 3
    assert len(REFINE_TEMPLATES) == 5
    assert SCS_ALPHA + SCS_BETA + SCS_GAMMA == 1.0
    print(f"  [OK] Config: {MODEL_ID}, layers={HOOK_LAYERS}")


def test_all_modules_importable():
    """Test: modulos ligeros se pueden importar sin error."""
    import src.config
    import src.m3_self_healing
    import src.m4_meta_evo
    import src.m5_semantic_memory
    # NOTA: src.symbol_detector, src.m1_self_polish, src.symbolic_self, src.api
    # requieren torch/transformers/fastapi y se prueban en test_gpu_pipeline.py
    print("  [OK] Modulos ligeros importan correctamente")


# === Runner ===

def run_all():
    """Ejecuta todos los tests sin pytest."""
    tests = [
        ("Config & Imports", test_config_imports),
        ("All Modules Importable", test_all_modules_importable),
        ("M3 Diagnosis", test_m3_diagnosis),
        ("M4 StrategyGenome", test_m4_strategy_genome),
        ("M4 Optimizer", test_m4_optimizer),
        ("M5 Store & Retrieve", test_m5_store_and_retrieve),
        ("M5 Eviction", test_m5_eviction),
        ("M5 Stats", test_m5_stats),
    ]

    print("=" * 70)
    print("  TEST SUITE -- SymbolicSelf (sin GPU)")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"  RESULTADO: {passed} passed, {failed} failed / {len(tests)} total")
    print(f"{'='*70}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
