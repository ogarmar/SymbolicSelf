# src/__init__.py
"""
SymbolicSelf — Self-correcting symbolic reasoning for Vision-Language Models.

Módulos:
    - symbol_detector: M2 — Emergent Symbol Detector (hooks + PCA + HDBSCAN + SCS)
    - m1_self_polish: M1 — Self-Polishing Core (variantes + selección por SCS)
    - m3_self_healing: M3 — Self-Healing Diagnostician (adversarial/drift detection)
    - symbolic_self: Pipeline maestro que integra M1 + M2 + M3
    - config: Configuración centralizada (rutas, hiperparámetros, constantes)
"""

from src.config import MODEL_ID
from src.m1_self_polish import SelfPolishCore
from src.m3_self_healing import DegradationType, Diagnosis, SelfHealingEngine
from src.symbol_detector import SymbolDetector
from src.symbolic_self import SymbolicResult, SymbolicSelfPipeline

__all__ = [
    "MODEL_ID",
    "SymbolDetector",
    "SelfPolishCore",
    "SelfHealingEngine",
    "DegradationType",
    "Diagnosis",
    "SymbolicSelfPipeline",
    "SymbolicResult",
]
