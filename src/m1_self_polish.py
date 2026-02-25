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
    from PIL import Image
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
        question: str,
        baseline_response: str = "",
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        n_variants: int = DEFAULT_N_VARIANTS,
    ) -> list[str]:
        """Genera N variantes refinadas.

        Cuando hay pixel_values, construye un prompt multimodal con <image>
        para cada variante (evitando el bug de orphaned pixel_values).
        Sin imagen, usa solo el language_model interno.
        """
        variants: list[str] = []

        for i, template in enumerate(self.templates[:n_variants]):
            if baseline_response:
                instruction = template.format(response=baseline_response)
            else:
                instruction = question

            # Construir prompt adecuado según modalidad
            if pixel_values is not None:
                # Multimodal: DEBE incluir <image> para que LLaVA expanda tokens
                refine_prompt = f"USER: <image>\n{instruction} ASSISTANT:"
                inputs = self.tokenizer(refine_prompt, return_tensors="pt").to(self.device)
                generate_kwargs = {
                    **inputs,
                    "pixel_values": pixel_values,
                    "max_new_tokens": GENERATION_MAX_TOKENS,
                    "do_sample": True,
                    "temperature": GENERATION_TEMPERATURE,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                if image_sizes is not None:
                    generate_kwargs["image_sizes"] = image_sizes
                with torch.no_grad():
                    output_ids = self.model.generate(**generate_kwargs)
            else:
                # Solo texto: usar language_model directamente
                inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": GENERATION_MAX_TOKENS,
                    "do_sample": True,
                    "temperature": GENERATION_TEMPERATURE,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                with torch.no_grad():
                    output_ids = self.model.language_model.generate(**generate_kwargs)

            # Decodificar solo tokens nuevos
            new_token_ids = output_ids[0][inputs.input_ids.shape[-1]:]
            variant_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            variants.append(variant_text)

            logger.info("V%d: %s", i + 1, variant_text[:100])

            # Liberar KV-cache entre variantes (6GB GPU)
            torch.cuda.empty_cache()

        return variants

    # ── Selección por SCS ──────────────────────────────────────────────────

    def select_best(
        self,
        question: str,
        variants: list[str],
        baseline_symbols,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
    ) -> tuple[str, float, dict]:
        """Evalúa todas las variantes y devuelve la mejor por SCS.

        Extrae símbolos de cada variante usando el prompt multimodal completo
        (con imagen si está disponible), para que las distribuciones de clusters
        sean comparables con los baseline_symbols.
        """
        best_scs = -1.0
        best_variant = ""
        best_metrics: dict = {}

        for i, variant_text in enumerate(variants):
            if pixel_values is not None:
                # Multimodal: incluir imagen en extracción de símbolos
                var_prompt = f"USER: <image>\n{variant_text} ASSISTANT:"
                var_ids = self.tokenizer(var_prompt, return_tensors="pt").input_ids.to(self.device)
                var_symbols, _, _ = self.detector.extract_symbols(
                    var_ids,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                )
            else:
                # Solo texto
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

            torch.cuda.empty_cache()

        return best_variant, best_scs, best_metrics

    # ── Pipeline completo ──────────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
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
        with torch.no_grad():
            if pixel_values is not None:
                generate_kwargs["pixel_values"] = pixel_values
                if image_sizes is not None:
                    generate_kwargs["image_sizes"] = image_sizes
                baseline_ids = self.model.generate(**generate_kwargs)
            else:
                baseline_ids = self.model.language_model.generate(**generate_kwargs)

        new_ids = baseline_ids[0][inputs.input_ids.shape[-1]:]
        baseline_response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        logger.info("Baseline: %s", baseline_response[:120])

        # 2. Extraer símbolos baseline (con imagen si disponible)
        baseline_symbols, _, _ = self.detector.extract_symbols(
            inputs.input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        torch.cuda.empty_cache()

        # Extraer la pregunta original del prompt para variantes
        # Formato: "USER: <image>\n{question} ASSISTANT:"
        question = prompt
        if "USER:" in prompt and "ASSISTANT:" in prompt:
            question = prompt.split("USER:")[-1].split("ASSISTANT:")[0].strip()
            question = question.replace("<image>", "").strip()

        # 3. Generar variantes
        variants = self.generate_variants(
            question,
            baseline_response=baseline_response,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            n_variants=n_variants,
        )

        # 4. Seleccionar la mejor
        if not variants:
            logger.warning("No se generaron variantes — devolviendo baseline.")
            return baseline_response, 0.0, {}

        best_text, best_scs, best_metrics = self.select_best(
            question, variants, baseline_symbols,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        # Si ninguna variante supera el baseline, devolver baseline
        if not best_text:
            return baseline_response, 0.0, {}

        logger.info("Ganadora: SCS=%.3f | %s", best_scs, best_text[:100])
        return best_text, best_scs, best_metrics
