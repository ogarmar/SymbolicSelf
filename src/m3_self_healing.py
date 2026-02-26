# src/m3_self_healing.py — MÓDULO 3: Self-Healing Diagnostician
"""
Detecta degradación del modelo (ataques adversariales o concept drift)
analizando los símbolos emergentes y aplica estrategias de healing.

Pipeline:
    1. Establecer baseline simbólico (estado "saludable")
    2. Comparar nuevos símbolos vs baseline → diagnóstico dual:
       - Adversarial: caída brusca de estabilidad + alta entropía
       - Drift: cambio gradual en distribución de clusters
    3. Acciones de healing según diagnóstico

Referencia: Self-Healing Machine Learning (SHML) paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.config import (
    ADVERSARIAL_STABILITY_THRESHOLD,
    DRIFT_STABILITY_THRESHOLD,
    ENTROPY_CHANGE_THRESHOLD,
)

if TYPE_CHECKING:
    from src.symbol_detector import SymbolDetector

logger = logging.getLogger(__name__)


class DegradationType(Enum):
    """Tipos de degradación detectables."""
    HEALTHY = "healthy"
    ADVERSARIAL = "adversarial"
    DRIFT = "drift"


@dataclass
class Diagnosis:
    """Resultado de un diagnóstico de degradación."""
    status: DegradationType
    stability: float
    entropy_delta: float
    healing_action: str

    def __str__(self) -> str:
        icons = {
            DegradationType.HEALTHY: "✅",
            DegradationType.ADVERSARIAL: "🦠",
            DegradationType.DRIFT: "📉",
        }
        return (
            f"{icons[self.status]} {self.status.value.upper()} | "
            f"stability={self.stability:.3f} entropy_Δ={self.entropy_delta:.3f} | "
            f"action: {self.healing_action}"
        )


class SelfHealingEngine:
    """Motor de auto-diagnóstico y healing para el pipeline SymbolicSelf.

    Compara símbolos actuales contra un baseline saludable para detectar
    degradación por ataques adversariales o concept drift.
    """

    def __init__(self, detector: SymbolDetector) -> None:
        self.detector = detector
        self.baseline_symbols: np.ndarray | None = None
        self.history: list[Diagnosis] = []

    # ── Baseline ───────────────────────────────────────────────────────────

    def establish_baseline(self, symbols: np.ndarray) -> np.ndarray:
        """Establece los símbolos de referencia (estado saludable).

        Args:
            symbols: Array de cluster IDs extraídos de una respuesta conocida.

        Returns:
            Los mismos símbolos (para encadenamiento).
        """
        self.baseline_symbols = symbols.copy()
        n_unique = len(set(symbols[symbols >= 0]))
        logger.info("Baseline establecido: %d símbolos únicos.", n_unique)
        return self.baseline_symbols

    # ── Diagnóstico ────────────────────────────────────────────────────────

    def diagnose(self, current_symbols: np.ndarray) -> Diagnosis:
        """Diagnóstico dual: adversarial vs drift vs healthy.

        Args:
            current_symbols: Símbolos de la respuesta actual a evaluar.

        Returns:
            Diagnosis con tipo, métricas y acción recomendada.
        """
        if self.baseline_symbols is None:
            raise RuntimeError(
                "Llama a establish_baseline() antes de diagnosticar."
            )

        # ── Estabilidad (1 - JSD baseline <-> actual) ─────────────────────
        dist_base = self._symbol_distribution(self.baseline_symbols)
        dist_curr = self._symbol_distribution(current_symbols)
        p, q = self._align_distributions(dist_base, dist_curr)

        from scipy.spatial.distance import jensenshannon
        jsd_val = float(jensenshannon(p, q))
        stability = 1.0 - jsd_val

        # ── Cambio de entropía ─────────────────────────────────────────
        entropy_baseline = self._compute_entropy(self.baseline_symbols)
        entropy_current = self._compute_entropy(current_symbols)
        entropy_delta = entropy_current - entropy_baseline

        # ── Clasificación ──────────────────────────────────────────────
        if stability < ADVERSARIAL_STABILITY_THRESHOLD or entropy_delta > ENTROPY_CHANGE_THRESHOLD:
            diagnosis = Diagnosis(
                status=DegradationType.ADVERSARIAL,
                stability=stability,
                entropy_delta=entropy_delta,
                healing_action="Purificación manifold (InfoNCE) + reducir temperatura",
            )
        elif stability < DRIFT_STABILITY_THRESHOLD:
            diagnosis = Diagnosis(
                status=DegradationType.DRIFT,
                stability=stability,
                entropy_delta=entropy_delta,
                healing_action="LoRA incremental + symbol-preserving loss",
            )
        else:
            diagnosis = Diagnosis(
                status=DegradationType.HEALTHY,
                stability=stability,
                entropy_delta=entropy_delta,
                healing_action="Continuar self-polish normalmente",
            )

        self.history.append(diagnosis)
        logger.info("Diagnostico: %s", diagnosis)
        return diagnosis

    # ── Utilidades de distribucion (evita circular import con SymbolDetector) ──

    @staticmethod
    def _symbol_distribution(symbols: np.ndarray) -> np.ndarray:
        """Convierte cluster IDs en distribucion de probabilidad normalizada."""
        valid = symbols[symbols >= 0]
        if len(valid) == 0:
            return np.array([1.0])
        max_id = int(valid.max()) + 1
        counts = np.bincount(valid, minlength=max_id).astype(float)
        total = counts.sum()
        if total == 0:
            return np.array([1.0])
        return counts / total

    @staticmethod
    def _align_distributions(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Alinea dos distribuciones al mismo tamano (padding con 0)."""
        max_len = max(len(p), len(q))
        p_aligned = np.zeros(max_len)
        q_aligned = np.zeros(max_len)
        p_aligned[:len(p)] = p
        q_aligned[:len(q)] = q
        return p_aligned, q_aligned

    # ── Entropia simbolica ─────────────────────────────────────────────────

    @staticmethod
    def _compute_entropy(symbols: np.ndarray) -> float:
        """Calcula la entropía de Shannon de la distribución de símbolos.

        Excluye ruido (cluster -1). Una entropía alta indica desorganización
        en las activaciones internas → posible ataque adversarial.
        """
        valid = symbols[symbols >= 0]
        if len(valid) == 0:
            return 0.0

        _, counts = np.unique(valid, return_counts=True)
        probs = counts / counts.sum()  # ← FIX: normalizar por total, no por nº categorías
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    # ── Simulación (para testing) ──────────────────────────────────────────

    def simulate_degradation(
        self,
        degradation_type: str = "drift",
        severity: float = 0.8,
    ) -> np.ndarray:
        """Genera símbolos degradados artificialmente para testing.

        Args:
            degradation_type: "adversarial" o "drift".
            severity: Intensidad de la degradación (0.0 a 2.0).

        Returns:
            Array de símbolos degradados.
        """
        if self.baseline_symbols is None:
            raise RuntimeError("Establece baseline primero.")

        rng = np.random.default_rng(42)
        n_unique = len(set(self.baseline_symbols[self.baseline_symbols >= 0]))
        n_unique = max(n_unique, 2)  # Evitar división por 0

        if degradation_type == "adversarial":
            # Reemplazar gran parte de los símbolos con IDs completamente distintos
            degraded = self.baseline_symbols.copy()
            n_to_flip = int(len(degraded) * min(severity, 1.0) * 0.8)
            flip_indices = rng.choice(len(degraded), size=n_to_flip, replace=False)
            # Asignar IDs fuera del rango original → Jaccard/JSD bajan
            new_ids = rng.integers(n_unique + 5, n_unique + 20, size=n_to_flip)
            degraded[flip_indices] = new_ids
        else:
            # Drift: shift gradual proporcional al nº de clusters
            shift = rng.integers(1, max(2, n_unique // 2), size=self.baseline_symbols.shape)
            degraded = self.baseline_symbols + shift

        return np.clip(degraded, -1, max(n_unique + 20, degraded.max()))
