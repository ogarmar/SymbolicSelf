# src/m1_self_polish.py — MÓDULO 1: Self-Polishing Core
"""
Genera N variantes refinadas de una respuesta base usando templates de
refinamiento y selecciona la mejor según SCS (Symbolic Coherence Score).

Pipeline:
    1. Respuesta baseline (LLaVA directo)
    2. Para cada template: generar variante refinada
    3. Extraer símbolos de cada variante via SymbolDetector
    4. Calcular SCS vs baseline → seleccionar argmax(SCS)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.config import (
    DEFAULT_N_VARIANTS,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    REFINE_TEMPLATES,
)
from src.symbol_detector import SymbolDetector

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SelfPolishCore:
    """Genera variantes refinadas y selecciona la mejor por SCS.

    Attributes:
        model: Modelo LLaVA cargado.
        tokenizer: Tokenizer asociado al modelo.
        detector: SymbolDetector para extraer símbolos y calcular SCS.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        detector: SymbolDetector,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.templates = REFINE_TEMPLATES
        self.device = next(model.parameters()).device

    # ── Generación de variantes ────────────────────────────────────────────

    def generate_variants(
        self,
        prompt: str,
        baseline_response: str = "",
        pixel_values: torch.Tensor | None = None,
        n_variants: int = DEFAULT_N_VARIANTS,
    ) -> list[str]:
        """Genera N variantes refinadas a partir de un prompt.

        Args:
            prompt: Pregunta original del usuario.
            baseline_response: Respuesta base a refinar. Si vacío, usa prompt directo.
            pixel_values: Tensor de imagen para modelos multimodales.
            n_variants: Número de variantes a generar.

        Returns:
            Lista de strings con las respuestas refinadas.
        """
        variants: list[str] = []

        for i, template in enumerate(self.templates[:n_variants]):
            if baseline_response:
                refine_prompt = template.format(response=baseline_response)
            else:
                refine_prompt = prompt

            inputs = self.tokenizer(refine_prompt, return_tensors="pt").to(self.device)

            generate_kwargs = {
                **inputs,
                "max_new_tokens": GENERATION_MAX_TOKENS,
                "do_sample": True,
                "temperature": GENERATION_TEMPERATURE,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if pixel_values is not None:
                generate_kwargs["pixel_values"] = pixel_values

            with torch.no_grad():
                output_ids = self.model.generate(**generate_kwargs)

            # Solo decodificar tokens nuevos (no el prompt de entrada)
            new_token_ids = output_ids[0][inputs.input_ids.shape[-1]:]
            variant_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            variants.append(variant_text)

            logger.info("V%d: %s", i + 1, variant_text[:100])

        return variants

    # ── Selección por SCS ──────────────────────────────────────────────────

    def select_best(
        self,
        prompt: str,
        variants: list[str],
        baseline_symbols,
    ) -> tuple[str, float, dict]:
        """Evalúa todas las variantes y devuelve la mejor por SCS.

        Args:
            prompt: Prompt original (para contextualizar los tokens).
            variants: Lista de respuestas candidatas.
            baseline_symbols: Símbolos extraídos de la respuesta baseline.

        Returns:
            (best_variant_text, best_scs_score, best_metrics)
        """
        best_scs = -1.0
        best_variant = ""
        best_metrics: dict = {}

        for i, variant_text in enumerate(variants):
            var_prompt = f"Refined answer: {variant_text}"
            var_ids = self.tokenizer(var_prompt, return_tensors="pt").input_ids.to(self.device)

            var_symbols, _, _ = self.detector.extract_symbols(var_ids)

            if len(var_symbols) == 0:
                logger.warning("V%d: No se extrajeron símbolos — skip.", i + 1)
                continue

            scs, metrics = self.detector.compute_scs(var_symbols, baseline_symbols)
            logger.info("V%d SCS=%.3f | %s", i + 1, scs, metrics)

            if scs > best_scs:
                best_scs = scs
                best_variant = variant_text
                best_metrics = metrics

        return best_variant, best_scs, best_metrics

    # ── Pipeline completo ──────────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        pixel_values: torch.Tensor | None = None,
        n_variants: int = DEFAULT_N_VARIANTS,
    ) -> tuple[str, float, dict]:
        """Pipeline completo: baseline → variantes → SCS → selección.

        Returns:
            (best_response, scs_score, metrics)
        """
        logger.info("Self-Polish: generando baseline para '%s'", prompt[:60])

        # 1. Baseline
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_kwargs = {
            **inputs,
            "max_new_tokens": GENERATION_MAX_TOKENS,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if pixel_values is not None:
            generate_kwargs["pixel_values"] = pixel_values

        with torch.no_grad():
            baseline_ids = self.model.generate(**generate_kwargs)

        new_ids = baseline_ids[0][inputs.input_ids.shape[-1]:]
        baseline_response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        logger.info("Baseline: %s", baseline_response[:120])

        # 2. Extraer símbolos baseline
        baseline_symbols, _, _ = self.detector.extract_symbols(inputs.input_ids)

        # 3. Generar variantes
        variants = self.generate_variants(
            prompt,
            baseline_response=baseline_response,
            pixel_values=pixel_values,
            n_variants=n_variants,
        )

        # 4. Seleccionar la mejor
        if not variants:
            logger.warning("No se generaron variantes — devolviendo baseline.")
            return baseline_response, 0.0, {}

        best_text, best_scs, best_metrics = self.select_best(
            prompt, variants, baseline_symbols,
        )

        # Si ninguna variante supera el baseline, devolver baseline
        if not best_text:
            return baseline_response, 0.0, {}

        logger.info("Ganadora: SCS=%.3f | %s", best_scs, best_text[:100])
        return best_text, best_scs, best_metrics
