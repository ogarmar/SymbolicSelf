# src/symbolic_self.py — Pipeline Maestro SymbolicSelf
"""
Orquesta los modulos M1 (Self-Polish), M2 (Symbol Detector),
M3 (Self-Healing) y M5 (Semantic Memory) en un pipeline secuencial:

    Input -> M5(retrieve) -> M1(variantes) -> M2(SCS) -> M3(healing) -> M5(store) -> Output
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import DEFAULT_N_VARIANTS, MODEL_ID
from src.m1_self_polish import SelfPolishCore
from src.m3_self_healing import SelfHealingEngine
from src.m5_semantic_memory import SemanticMemory
from src.model_loader import load_model_sync
from src.symbol_detector import SymbolDetector

logger = logging.getLogger(__name__)

# Evitar fragmentacion de VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Typed dataclasses ──────────────────────────────────────────────────

@dataclass
class SCSMetrics:
    """Metricas individuales del Symbolic Coherence Score."""
    consistency: float = 0.0
    stability: float = 0.0
    cross_modal: float = 0.0


@dataclass
class SymbolicResult:
    """Resultado completo de una invocacion del pipeline."""
    response: str
    scs: float
    metrics: SCSMetrics = field(default_factory=SCSMetrics)
    diagnosis: str = "healthy"
    healing_action: str = ""
    latency_ms: float = 0.0


class SymbolicSelfPipeline:
    """Pipeline maestro que integra M1 + M2 + M3 + M5.

    Uso:
        pipeline = SymbolicSelfPipeline()
        result = pipeline.process("Describe this image", image)
    """

    def __init__(self, model_id: str = MODEL_ID, adapter_path: str | None = None) -> None:
        """Inicializa pipeline usando model_loader centralizado.

        FIX 1: Usa load_model_sync() en vez de cargar el modelo inline.
        Esto evita tener dos modelos en VRAM si se crea mas de una instancia.
        """
        start = time.time()

        # ── Carga delegada al singleton centralizado ───────────────────
        self.model, self.processor = load_model_sync(
            model_id=model_id,
            adapter_path=adapter_path,
        )

        self.detector = SymbolDetector(self.model)
        self.polisher = SelfPolishCore(
            self.model, self.processor.tokenizer, self.detector,
        )
        self.healer = SelfHealingEngine(self.detector)
        self.memory = SemanticMemory(max_entries=500, embedding_dim=4096)
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
            n_variants: Numero de variantes para self-polish.

        Returns:
            SymbolicResult con respuesta, SCS, diagnostico y metricas.
        """
        start = time.time()

        # ── Preparar inputs ────────────────────────────────────────────
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pixel_values = inputs.get("pixel_values")
        image_sizes = inputs.get("image_sizes")

        # ── M1: Self-Polish (baseline + variantes + seleccion por SCS) ─
        best_response, best_scs, raw_metrics = self.polisher.run(
            prompt,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            n_variants=n_variants,
        )

        # Convertir dict -> SCSMetrics tipado
        metrics = SCSMetrics(
            consistency=raw_metrics.get("consistency", 0.0),
            stability=raw_metrics.get("stability", 0.0),
            cross_modal=raw_metrics.get("cross_modal", 0.0),
        )

        # ── M2: Extraer simbolos finales (incluyendo imagen) ──────────
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

        # Actualizar referencia de estabilidad explicitamente
        if len(final_symbols) > 0:
            self.detector.update_reference(final_symbols)

        # ── M3: Diagnostico de degradacion ─────────────────────────────
        diagnosis_str = "healthy"
        healing_action = ""
        if len(final_symbols) > 0:
            if not self._baseline_established:
                self.healer.establish_baseline(final_symbols)
                self._baseline_established = True
                diagnosis_str = "baseline_set"
            else:
                diagnosis = self.healer.diagnose(final_symbols)
                diagnosis_str = diagnosis.status.value
                healing_action = diagnosis.healing_action

        # ── M5: Almacenar en memoria semantica ─────────────────────────
        if len(final_symbols) > 0:
            embedding = np.zeros(4096, dtype=np.float32)
            last_layer_key = f"layer_{max(self.detector.layers)}"
            if last_layer_key in self.detector.activations:
                act = self.detector.activations[last_layer_key]
                embedding = act.reshape(-1, act.shape[-1]).mean(axis=0)[:4096]

            self.memory.store(
                question=question,
                answer=best_response,
                embedding=embedding,
                symbols=final_symbols,
                scs=best_scs,
            )

        # ── FIX 4: Limpiar activaciones tras almacenar ─────────────────
        self.detector.activations.clear()

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
