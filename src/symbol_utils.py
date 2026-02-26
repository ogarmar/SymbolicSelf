# src/symbol_utils.py — Utilidades compartidas para distribuciones simbolicas
"""
Funciones de distribucion y alineamiento usadas por SymbolDetector (M2)
y SelfHealingEngine (M3). Extraidas para romper la dependencia circular.

FIX 7: DRY — estas funciones estaban duplicadas en symbol_detector.py
y m3_self_healing.py.
"""

from __future__ import annotations

import numpy as np


def symbol_distribution(symbols: np.ndarray) -> np.ndarray:
    """Convierte cluster IDs en distribucion de probabilidad normalizada.

    Excluye ruido (-1). Devuelve vector de frecuencias relativas
    indexado por cluster ID.

    Args:
        symbols: Array de cluster IDs (HDBSCAN output).

    Returns:
        Array de probabilidades normalizadas.
    """
    valid = symbols[symbols >= 0]
    if len(valid) == 0:
        return np.array([1.0])

    max_id = int(valid.max()) + 1
    counts = np.bincount(valid, minlength=max_id).astype(float)
    total = counts.sum()
    if total == 0:
        return np.array([1.0])
    return counts / total


def align_distributions(
    p: np.ndarray, q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Alinea dos distribuciones al mismo tamano (padding con 0).

    Necesario para Jensen-Shannon divergence, que requiere
    distribuciones de la misma longitud.

    Args:
        p: Primera distribucion.
        q: Segunda distribucion.

    Returns:
        Tupla (p_aligned, q_aligned) con misma longitud.
    """
    max_len = max(len(p), len(q))
    p_aligned = np.zeros(max_len)
    q_aligned = np.zeros(max_len)
    p_aligned[:len(p)] = p
    q_aligned[:len(q)] = q
    return p_aligned, q_aligned
