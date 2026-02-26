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
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

from src.config import (
    HDBSCAN_MIN_CLUSTER,
    HDBSCAN_MIN_SAMPLES,
    HOOK_LAYERS,
    MAX_TOKENS_PCA,
    PCA_COMPONENTS_TOKENS,
    SCS_ALPHA,
    SCS_BETA,
    SCS_GAMMA,
)
from src.symbol_utils import align_distributions, symbol_distribution

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

        # Hook en vision tower para cross-modal alignment
        self._register_vision_hook()

        logger.info("Hooks registrados en %d capas (incl. vision).", len(self.hooks))

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
        """Crea un forward-hook closure que captura hidden states.

        Maneja múltiples tipos de output:
          - tuple → output[0] (capas LLM)
          - ModelOutput con .last_hidden_state (vision tower)
          - tensor directo
        """
        def hook_fn(module, input_, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            else:
                hidden = output
            self.activations[layer_name] = hidden.detach().cpu().numpy()
        return hook_fn

    def _register_vision_hook(self) -> None:
        """Registra hook en la salida de la vision tower para cross-modal."""
        vision_tower = None
        for name, module in self.model.named_modules():
            if name == "model.vision_tower":
                vision_tower = module
                break
        if vision_tower is None:
            # Buscar por atributo directo
            if hasattr(self.model, "vision_tower"):
                vision_tower = self.model.vision_tower
            elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
                vision_tower = self.model.model.vision_tower

        if vision_tower is not None:
            hook = vision_tower.register_forward_hook(self._make_capture_fn("vision_out"))
            self.hooks.append(hook)
            logger.info("Vision tower hook registrado.")
        else:
            logger.warning("No se encontró vision tower — cross-modal usará fallback.")

    # ── Symbol extraction ──────────────────────────────────────────────────

    def extract_symbols(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        **model_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Ejecuta forward pass y extrae símbolos emergentes.

        Args:
            input_ids: Token IDs (batch, seq_len).
            pixel_values: Tensor de imagen opcional para modelos multimodales.
            **model_kwargs: Parámetros extra para el forward del modelo
                            (e.g. image_sizes para LLaVA-Next).

        Returns:
            (symbols, latent, variance_retained):
                - symbols: array int con cluster ID por token (-1 = ruido)
                - latent: representación PCA global (batch, components)
                - variance_retained: varianza retenida por PCA global
        """
        self.activations.clear()

        # Forward pass — activa los hooks
        # LLaVA-Next requiere pixel_values + image_sizes + attention_mask
        try:
            with torch.no_grad():
                if pixel_values is not None:
                    # attention_mask es obligatorio para _merge_input_ids_with_image_features
                    if "attention_mask" not in model_kwargs:
                        model_kwargs["attention_mask"] = torch.ones_like(input_ids)
                    self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        **model_kwargs,
                    )
                else:
                    # Forward por el language model (Mistral) directamente
                    self.model.language_model(input_ids=input_ids)
        except Exception:
            self.activations.clear()
            torch.cuda.empty_cache()
            raise

        if not self.activations:
            logger.warning("No se capturaron activaciones — hooks registrados correctamente?")
            return np.array([]), np.array([]), 0.0

        # ── Fusión multi-capa a nivel de TOKEN ─────────────────────────
        # Concatenar activaciones de las N capas por token:
        #   layer_12: (batch, seq, hidden), layer_15: ..., layer_18: ...
        #   → fusión: (batch, seq, N_layers * hidden)
        #   → aplanar batch: (total_tokens, N_layers * hidden)
        available_layers = [f"layer_{i}" for i in self.layers if f"layer_{i}" in self.activations]
        acts_per_layer = [self.activations[name] for name in available_layers]

        # Recortar al seq_len mínimo (por si difieren entre capas)
        min_seq = min(a.shape[1] for a in acts_per_layer)
        acts_trimmed = [a[:, :min_seq, :] for a in acts_per_layer]

        # Concatenar por hidden dim: (batch, seq, N*hidden)
        acts_fused = np.concatenate(acts_trimmed, axis=-1)

        # Aplanar batch×seq → tokens: (n_tokens, fused_dim)
        total_tokens = acts_fused.reshape(-1, acts_fused.shape[-1]).shape[0]
        if total_tokens > MAX_TOKENS_PCA:
            logger.warning(
                "Truncando %d tokens a MAX_TOKENS_PCA=%d para PCA. "
                "Tokens de imagen AnyRes descartados silenciosamente.",
                total_tokens, MAX_TOKENS_PCA,
            )
        token_flat = acts_fused.reshape(-1, acts_fused.shape[-1])[:MAX_TOKENS_PCA]
        n_tokens = token_flat.shape[0]

        # ── PCA sobre tokens ───────────────────────────────────────────
        n_comp = min(PCA_COMPONENTS_TOKENS, n_tokens - 1)  # PCA necesita n > n_comp
        if n_comp < 2:
            logger.warning("Muy pocos tokens (%d) para PCA.", n_tokens)
            return np.zeros(n_tokens, dtype=int), token_flat, 0.0

        pca = PCA(n_components=n_comp)
        token_latent = pca.fit_transform(token_flat)
        variance_retained = float(pca.explained_variance_ratio_.sum())

        # FIX 8: Advertencia si PCA retiene poca varianza
        if variance_retained < 0.15:
            logger.warning(
                "PCA retiene solo %.1f%% de varianza (<15%%). "
                "El clustering puede ser poco fiable.",
                variance_retained * 100,
            )

        # ── HDBSCAN con min_cluster_size dinámico ──────────────────────
        # Escalar con n_tokens: para 1400 tokens → ~28 → produce 5-15 clusters
        dynamic_min_cluster = max(10, n_tokens // 50)
        min_cluster_size = max(dynamic_min_cluster, HDBSCAN_MIN_CLUSTER)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_method="leaf",  # Clusters más homogéneos
        )
        symbols = clusterer.fit_predict(token_latent)

        n_symbols = len(set(symbols[symbols >= 0]))
        n_noise = int((symbols == -1).sum())
        logger.info(
            "%d símbolos emergentes, %d ruido, %d tokens (var PCA: %.1f%%)",
            n_symbols, n_noise, n_tokens, variance_retained * 100,
        )

        return symbols, token_latent, variance_retained

    # ── Symbolic Coherence Score ───────────────────────────────────────────

    def compute_scs(
        self,
        symbols_current: np.ndarray,
        symbols_baseline: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Calcula SCS = α·consistency + β·stability + γ·cross_modal.

        Usa Jensen-Shannon divergence sobre distribuciones de símbolos
        en vez de Jaccard sobre sets de IDs.

        Args:
            symbols_current: Clusters de la variante actual.
            symbols_baseline: Clusters de la respuesta baseline.

        Returns:
            (scs_score, metrics_dict) con keys consistency/stability/cross_modal.
        """
        # ── Consistency (1 - JSD entre current y baseline) ─────────────
        dist_curr = symbol_distribution(symbols_current)
        dist_base = symbol_distribution(symbols_baseline)
        p, q = align_distributions(dist_curr, dist_base)
        jsd = jensenshannon(p, q)  # 0 = idénticas, 1 = totalmente distintas
        consistency = 1.0 - float(jsd)

        # ── Stability (1 - JSD entre current y extracción anterior) ────
        if self._previous_symbols is not None and len(self._previous_symbols) > 0:
            dist_prev = symbol_distribution(self._previous_symbols)
            p2, q2 = align_distributions(dist_curr, dist_prev)
            stability = 1.0 - float(jensenshannon(p2, q2))
        else:
            stability = 1.0  # Primera iteración

        # ── Cross-modal: cosine similarity vision vs language ────────
        if "vision_out" in self.activations and f"layer_{max(self.layers)}" in self.activations:
            vis_act = self.activations["vision_out"]
            lang_act = self.activations[f"layer_{max(self.layers)}"]
            # Mean pooling across tokens
            vis_mean = vis_act.reshape(-1, vis_act.shape[-1]).mean(axis=0)
            lang_mean = lang_act.reshape(-1, lang_act.shape[-1]).mean(axis=0)
            # Las dimensiones pueden diferir (vision vs language hidden)
            # Truncar al mínimo
            min_dim = min(len(vis_mean), len(lang_mean))
            vis_vec = vis_mean[:min_dim]
            lang_vec = lang_mean[:min_dim]
            # Cosine similarity
            dot = np.dot(vis_vec, lang_vec)
            norm_v = np.linalg.norm(vis_vec) + 1e-8
            norm_l = np.linalg.norm(lang_vec) + 1e-8
            cross_modal = float(np.clip(dot / (norm_v * norm_l), 0.0, 1.0))
        else:
            # Fallback: usar ratio de tokens asignados (normalizado)
            logger.warning(
                "Vision hook no activo — cross_modal usa fallback. "
                "El componente cross-modal del SCS no es fiable para esta muestra."
            )
            n_assigned = int((symbols_current >= 0).sum())
            n_total = len(symbols_current)
            raw_ratio = n_assigned / max(n_total, 1)
            cross_modal = 1.0 / (1.0 + np.exp(-10 * (raw_ratio - 0.3)))

        # NOT updated here — use update_reference() explicitly from pipeline

        scs = SCS_ALPHA * consistency + SCS_BETA * stability + SCS_GAMMA * cross_modal
        scs = min(1.0, max(0.0, scs))

        metrics = {
            "consistency": round(consistency, 4),
            "stability": round(stability, 4),
            "cross_modal": round(cross_modal, 4),
        }

        logger.info("SCS=%.3f | %s", scs, metrics)
        return scs, metrics

    def update_reference(self, symbols: np.ndarray) -> None:
        """Actualiza la referencia de estabilidad explicitamente.

        Llamar desde el pipeline despues de seleccionar la mejor variante.
        NO se actualiza automaticamente en compute_scs.
        """
        self._previous_symbols = symbols.copy()

    # ── Cleanup ────────────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        """Elimina todos los hooks registrados."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("Hooks eliminados.")

    def __del__(self) -> None:
        self.remove_hooks()
