# src/symbol_detector.py — MÓDULO 2: Emergent Symbol Detector & Tracker
"""
Extrae símbolos emergentes de activaciones internas de LLaVA mediante:
  1. PyTorch hooks en capas 12, 15, 18 del LLM backbone
  2. PCA para reducción de dimensionalidad (4096 → 64/256)
  3. HDBSCAN para clustering sin k fijo → "símbolos emergentes"
  4. SCS (Symbolic Coherence Score) = α·consistency + β·stability + γ·alignment
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hdbscan
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.config import (
    HDBSCAN_MIN_CLUSTER,
    HDBSCAN_MIN_SAMPLES,
    HOOK_LAYERS,
    MAX_TOKENS_PCA,
    PCA_COMPONENTS_GLOBAL,
    PCA_COMPONENTS_TOKENS,
    SCS_ALPHA,
    SCS_BETA,
    SCS_GAMMA,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class SymbolDetector:
    """Detecta símbolos emergentes en las activaciones internas de un VLM.

    Resumen del pipeline:
        forward pass → hooks capturan activaciones → PCA → HDBSCAN → symbols + SCS
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        layers: list[int] | None = None,
    ) -> None:
        self.model = model
        self.layers = layers or HOOK_LAYERS
        self.activations: dict[str, np.ndarray] = {}
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self._previous_symbols: np.ndarray | None = None

        if self.model is not None:
            self._register_hooks()

        logger.info("SymbolDetector inicializado: layers %s", self.layers)

    # ── Hook registration ──────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Busca dinámicamente las capas del LLM backbone y registra hooks."""
        llm_layers = self._find_llm_layers()
        if llm_layers is None:
            logger.warning(
                "No se encontraron capas LLM — el detector funcionará en modo simulado."
            )
            return

        for idx in self.layers:
            if idx >= len(llm_layers):
                logger.warning("Capa %d fuera de rango (modelo tiene %d capas).", idx, len(llm_layers))
                continue
            hook = llm_layers[idx].register_forward_hook(self._make_capture_fn(f"layer_{idx}"))
            self.hooks.append(hook)

        logger.info("Hooks registrados en %d capas.", len(self.hooks))

    def _find_llm_layers(self) -> torch.nn.ModuleList | None:
        """Búsqueda dinámica: encuentra la ModuleList de capas del LLM (no visión)."""
        for name, module in self.model.named_modules():
            if (
                name.endswith("layers")
                and isinstance(module, torch.nn.ModuleList)
                and "vision" not in name
            ):
                logger.info("LLM backbone encontrado en: %s", name)
                return module
        return None

    def _make_capture_fn(self, layer_name: str):
        """Crea un forward-hook closure que captura hidden states."""
        def hook_fn(module, input_, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_name] = hidden.detach().cpu().numpy()
        return hook_fn

    # ── Symbol extraction ──────────────────────────────────────────────────

    def extract_symbols(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Ejecuta forward pass y extrae símbolos emergentes.

        Args:
            input_ids: Token IDs (batch, seq_len).
            pixel_values: Tensor de imagen opcional para modelos multimodales.

        Returns:
            (symbols, latent, variance_retained):
                - symbols: array int con cluster ID por token (-1 = ruido)
                - latent: representación PCA global (batch, components)
                - variance_retained: varianza retenida por PCA global
        """
        self.activations.clear()

        # Forward pass — activa los hooks
        forward_kwargs = {"input_ids": input_ids}
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values

        with torch.no_grad():
            self.model(**forward_kwargs)

        if not self.activations:
            logger.warning("No se capturaron activaciones — ¿hooks registrados correctamente?")
            return np.array([]), np.array([]), 0.0

        # ── PCA Global (fusión multi-capa) ─────────────────────────────
        available_layers = [f"layer_{i}" for i in self.layers if f"layer_{i}" in self.activations]
        acts_list = [self.activations[name].mean(axis=1) for name in available_layers]
        acts_fused = np.concatenate(acts_list, axis=-1)  # (batch, N_layers * hidden_dim)

        n_samples = acts_fused.shape[0]
        latent = acts_fused
        variance_retained = 0.0

        if n_samples > 1:
            n_comp = min(PCA_COMPONENTS_GLOBAL, n_samples)
            pca_global = PCA(n_components=n_comp)
            latent = pca_global.fit_transform(acts_fused)
            variance_retained = float(pca_global.explained_variance_ratio_.sum())

        # ── HDBSCAN en tokens de la capa central (layer_15) ────────────
        central_key = f"layer_{self.layers[len(self.layers) // 2]}"
        token_acts = self.activations[central_key]
        token_flat = token_acts.reshape(-1, token_acts.shape[-1])[:MAX_TOKENS_PCA]
        n_tokens = token_flat.shape[0]

        n_comp_tok = min(PCA_COMPONENTS_TOKENS, n_tokens)
        pca_tokens = PCA(n_components=n_comp_tok)
        token_latent = pca_tokens.fit_transform(token_flat)

        min_cluster_size = max(2, min(HDBSCAN_MIN_CLUSTER, n_tokens))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=HDBSCAN_MIN_SAMPLES,
        )
        symbols = clusterer.fit_predict(token_latent)

        n_symbols = len(set(symbols[symbols >= 0]))
        logger.info(
            "%d símbolos emergentes detectados en %d tokens (varianza PCA: %.1f%%)",
            n_symbols, n_tokens, pca_tokens.explained_variance_ratio_.sum() * 100,
        )

        return symbols, latent, variance_retained

    # ── Symbolic Coherence Score ───────────────────────────────────────────

    def compute_scs(
        self,
        symbols_current: np.ndarray,
        symbols_baseline: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Calcula SCS = α·consistency + β·stability + γ·cross_modal.

        Args:
            symbols_current: Clusters de la variante actual.
            symbols_baseline: Clusters de la respuesta baseline.

        Returns:
            (scs_score, metrics_dict)
        """
        # ── Consistency (Jaccard entre current y baseline) ─────────────
        set_curr = set(symbols_current[symbols_current >= 0])
        set_base = set(symbols_baseline[symbols_baseline >= 0])
        intersection = len(set_curr & set_base)
        union = len(set_curr | set_base)
        consistency = intersection / max(union, 1)

        # ── Stability (Jaccard entre current y la extracción anterior) ─
        if self._previous_symbols is not None and len(self._previous_symbols) > 0:
            set_prev = set(self._previous_symbols[self._previous_symbols >= 0])
            inter_prev = len(set_curr & set_prev)
            union_prev = len(set_curr | set_prev)
            stability = inter_prev / max(union_prev, 1)
        else:
            stability = 1.0  # Primera iteración: máxima estabilidad asumida

        # ── Cross-modal alignment ─────────────────────────────────────
        # TODO(M2): Implementar comparación real visual encoder vs text decoder
        # Por ahora usamos la consistencia como proxy (correlación alta empírica)
        cross_modal = consistency * 0.9 + 0.1

        # Guardar para próxima iteración
        self._previous_symbols = symbols_current.copy()

        scs = SCS_ALPHA * consistency + SCS_BETA * stability + SCS_GAMMA * cross_modal
        scs = min(1.0, max(0.0, scs))

        metrics = {
            "consistency": round(consistency, 4),
            "stability": round(stability, 4),
            "cross_modal": round(cross_modal, 4),
        }

        logger.info("SCS=%.3f | %s", scs, metrics)
        return scs, metrics

    # ── Cleanup ────────────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        """Elimina todos los hooks registrados."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("Hooks eliminados.")

    def __del__(self) -> None:
        self.remove_hooks()
