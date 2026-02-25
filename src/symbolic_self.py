# src/symbolic_self.py — Pipeline Maestro SymbolicSelf
"""
Orquesta los módulos M1 (Self-Polish), M2 (Symbol Detector) y M3 (Self-Healing)
en un pipeline secuencial completo:

    Input → M2(baseline) → M1(variantes) → M2(SCS) → M3(healing) → Output

Módulos futuros (M4 Meta-Evo, M5 Memory) se insertarán aquí cuando estén listos.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from src.config import (
    DEFAULT_N_VARIANTS,
    MAX_MEMORY,
    MODEL_ID,
    QUANTIZATION,
    TORCH_DTYPE,
)
from src.m1_self_polish import SelfPolishCore
from src.m3_self_healing import SelfHealingEngine
from src.symbol_detector import SymbolDetector

logger = logging.getLogger(__name__)

# Evitar fragmentación de VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class SymbolicResult:
    """Resultado completo de una invocación del pipeline."""
    response: str
    scs: float
    metrics: dict = field(default_factory=dict)
    diagnosis: str = "healthy"
    healing_action: str = ""
    latency_ms: float = 0.0


class SymbolicSelfPipeline:
    """Pipeline maestro que integra M1 + M2 + M3.

    Uso:
        pipeline = SymbolicSelfPipeline()
        result = pipeline.process("Describe this image", image)
    """

    def __init__(self, model_id: str = MODEL_ID, adapter_path: str | None = None) -> None:
        logger.info("Cargando modelo %s ...", model_id)
        start = time.time()

        bnb_config = BitsAndBytesConfig(**QUANTIZATION)
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
            max_memory=MAX_MEMORY,
            low_cpu_mem_usage=True,
        )

        # ── LoRA adapter (si existe) ───────────────────────────────────
        self.adapter_path = adapter_path
        if adapter_path:
            from pathlib import Path
            if Path(adapter_path).exists():
                from peft import PeftModel
                logger.info("Cargando LoRA adapter desde %s", adapter_path)
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                logger.info("LoRA adapter cargado — modelo fine-tuned activo.")
            else:
                logger.warning("adapter_path=%s no existe, usando base model.", adapter_path)

        self.detector = SymbolDetector(self.model)
        self.polisher = SelfPolishCore(
            self.model, self.processor.tokenizer, self.detector,
        )
        self.healer = SelfHealingEngine(self.detector)
        self._baseline_established = False

        logger.info("Pipeline listo en %.1fs.", time.time() - start)

    # ── Punto de entrada principal ─────────────────────────────────────────

    def process(
        self,
        question: str,
        image=None,
        n_variants: int = DEFAULT_N_VARIANTS,
    ) -> SymbolicResult:
        """Ejecuta el pipeline completo sobre una imagen + pregunta.

        Args:
            question: Pregunta VQA.
            image: PIL.Image o None (modo texto puro).
            n_variants: Número de variantes para self-polish.

        Returns:
            SymbolicResult con respuesta, SCS, diagnóstico y métricas.
        """
        start = time.time()

        # ── Preparar inputs ────────────────────────────────────────────
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pixel_values = inputs.get("pixel_values")
        image_sizes = inputs.get("image_sizes")

        # ── M1: Self-Polish (baseline + variantes + selección por SCS) ─
        best_response, best_scs, metrics = self.polisher.run(
            prompt,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            n_variants=n_variants,
        )

        # ── M2: Extraer símbolos finales (incluyendo imagen) ──────────────
        final_prompt = f"USER: <image>\n{question}\n{best_response} ASSISTANT:"
        final_inputs = self.processor(
            text=final_prompt, images=image, return_tensors="pt",
        )
        final_inputs = {k: v.to(self.model.device) for k, v in final_inputs.items()}
        final_symbols, _, _ = self.detector.extract_symbols(
            final_inputs["input_ids"],
            pixel_values=final_inputs.get("pixel_values"),
            image_sizes=final_inputs.get("image_sizes"),
        )

        # ── M3: Diagnóstico de degradación ─────────────────────────────
        diagnosis_str = "healthy"
        healing_action = ""
        if len(final_symbols) > 0:
            if not self._baseline_established:
                # Primera llamada: establecer referencia
                self.healer.establish_baseline(final_symbols)
                self._baseline_established = True
                diagnosis_str = "baseline_set"
            else:
                # Llamadas siguientes: diagnosticar contra la referencia
                diagnosis = self.healer.diagnose(final_symbols)
                diagnosis_str = diagnosis.status.value
                healing_action = diagnosis.healing_action

        # ── Limpiar VRAM ───────────────────────────────────────────────
        torch.cuda.empty_cache()

        elapsed_ms = (time.time() - start) * 1000

        return SymbolicResult(
            response=best_response,
            scs=best_scs,
            metrics=metrics,
            diagnosis=diagnosis_str,
            healing_action=healing_action,
            latency_ms=round(elapsed_ms, 1),
        )
