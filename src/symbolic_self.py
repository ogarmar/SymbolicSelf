# src/symbolic_self.py — Pipeline Maestro SymbolicSelf
"""
Orquesta los modulos M1 (Self-Polish), M2 (Symbol Detector),
M3 (Self-Healing), M4 (Meta-Evolutionary hyperparams) y
M5 (Semantic Memory) en un pipeline secuencial:

    Input -> M5(retrieve) -> M1(variantes) -> M2(SCS) -> M3(healing) -> M5(store) -> Output

NOTA: No usar con uvicorn --workers > 1.
El singleton de model_loader no comparte estado entre procesos.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import (
    DEFAULT_N_VARIANTS,
    GENERATION_TEMPERATURE,
    MAX_TOKENS_PCA,
    MODEL_ID,
    PROMPT_TEMPLATE,
)
from src.m1_self_polish import SelfPolishCore
from src.m3_self_healing import SelfHealingEngine
from src.m4_meta_evo import StrategyGenome
from src.m5_semantic_memory import SemanticMemory
from src.model_loader import load_model_sync
from src.symbol_detector import SymbolDetector

logger = logging.getLogger(__name__)

# Evitar fragmentacion de VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Minimo de clusters validos para aceptar un baseline
MIN_BASELINE_CLUSTERS = 2


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
    """Pipeline maestro que integra M1 + M2 + M3 + M4 + M5.

    Uso:
        pipeline = SymbolicSelfPipeline()
        result = pipeline.process("Describe this image", image)
    """

    def __init__(self, model_id: str = MODEL_ID, adapter_path: str | None = None) -> None:
        """Inicializa pipeline usando model_loader centralizado."""
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

        # M4: Usar StrategyGenome como fuente de hiperparametros.
        # Actualmente usa defaults; tras optimize() se puede pasar
        # el best_genome resultante.
        self._genome = StrategyGenome()

        logger.info("Pipeline listo en %.1fs.", time.time() - start)

    # ── Status (desacoplado de api.py) ─────────────────────────────────

    def get_status(self) -> dict:
        """Estado publico del pipeline para consumo por api.py."""
        return {
            "baseline_established": self._baseline_established,
            "memory_entries": len(self.memory),
            "genome": {
                "n_variants": self._genome.n_variants,
                "temperature": self._genome.temperature,
                "scs_weights": (
                    self._genome.scs_alpha,
                    self._genome.scs_beta,
                    self._genome.scs_gamma,
                ),
            },
        }

    # ── Punto de entrada principal ─────────────────────────────────────────

    def process(
        self,
        question: str,
        image=None,
        n_variants: int | None = None,
    ) -> SymbolicResult:
        """Ejecuta el pipeline completo sobre una imagen + pregunta.

        Args:
            question: Pregunta VQA.
            image: PIL.Image o None (modo texto puro).
            n_variants: Numero de variantes. Si None, usa el genome M4.

        Returns:
            SymbolicResult con respuesta, SCS, diagnostico y metricas.
        """
        # M4: usar hiperparametros del genome si no se especifican
        if n_variants is None:
            n_variants = self._genome.n_variants

        start = time.time()

        try:
            return self._process_impl(question, image, n_variants, start)
        finally:
            # Garantizar limpieza de activaciones y VRAM ante cualquier
            # excepcion mid-pipeline (hooks quedan registrados, pero
            # activaciones stale se eliminan).
            self.detector.activations.clear()
            torch.cuda.empty_cache()

    def _process_impl(
        self,
        question: str,
        image,
        n_variants: int,
        start: float,
    ) -> SymbolicResult:
        """Implementacion interna de process(), envuelta en try/finally."""

        # ── Preparar inputs para M5 retrieve ───────────────────────────
        # Hacemos un forward pass ligero con el prompt original para
        # extraer un embedding real que use como query de M5 retrieve.
        prompt = PROMPT_TEMPLATE.format(question=question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pixel_values = inputs.get("pixel_values")
        image_sizes = inputs.get("image_sizes")

        # ── M5: Recuperar contexto de memoria semantica ────────────────
        memory_hint = ""
        if len(self.memory) > 0:
            # Extraer embedding real del prompt para un retrieve preciso.
            # Esto requiere un forward pass, pero reutilizamos las
            # activaciones que los hooks ya capturan.
            query_symbols, _, _ = self.detector.extract_symbols(
                inputs["input_ids"],
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            # Construir query embedding desde activaciones del ultimo layer
            query_emb = np.zeros(4096, dtype=np.float32)
            last_layer_key = f"layer_{max(self.detector.layers)}"
            if last_layer_key in self.detector.activations:
                act = self.detector.activations[last_layer_key]
                query_emb = act.reshape(-1, act.shape[-1]).mean(axis=0)[:4096]

            past = self.memory.retrieve(query_emb, top_k=2, min_similarity=0.3)
            if past:
                memory_hint = "\nContext from memory: " + "; ".join(
                    f"[Q: {e.question[:40]} -> A: {e.answer[:40]}]"
                    for e, _ in past
                )
                logger.info("M5 retrieve: %d entradas relevantes.", len(past))

        # ── Reconstruir prompt con hint (si existe) ────────────────────
        if memory_hint:
            prompt = f"USER: <image>\n{question}{memory_hint} ASSISTANT:"
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
        # NOTE: Estos simbolos representan el estado interno del modelo al
        # procesar (question + best_response) juntos — NO el estado de
        # activaciones generativas durante la produccion de best_response.
        # Esta distincion se reconoce como limitacion del enfoque de
        # coherencia simbolica. Ver seccion correspondiente de la tesis.
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

        # Actualizar referencia de estabilidad explicitamente.
        # NOTA ARQUITECTONICA: Tras M1.run(), _previous_symbols apunta a
        # baseline_symbols (establecido dentro de run()). Aqui se actualiza
        # a final_symbols del tercer forward pass. En la siguiente llamada
        # a process(), la metrica de estabilidad comparara los nuevos
        # simbolos contra final_symbols de ESTA llamada. La cadena es:
        #   third-pass symbols(call N) -> stability baseline(call N+1)
        if len(final_symbols) > 0:
            self.detector.update_reference(final_symbols)

        # ── M3: Diagnostico de degradacion ─────────────────────────────
        diagnosis_str = "healthy"
        healing_action = ""
        if len(final_symbols) > 0:
            n_clusters = len(set(final_symbols[final_symbols >= 0]))
            if not self._baseline_established:
                if n_clusters >= MIN_BASELINE_CLUSTERS:
                    self.healer.establish_baseline(final_symbols)
                    self._baseline_established = True
                    diagnosis_str = "baseline_set"
                else:
                    logger.warning(
                        "Solo %d clusters validos (<MIN_BASELINE_CLUSTERS=%d), "
                        "baseline no establecido — posiblemente degenerate.",
                        n_clusters, MIN_BASELINE_CLUSTERS,
                    )
                    diagnosis_str = "baseline_deferred"
            else:
                diagnosis = self.healer.diagnose(final_symbols)
                diagnosis_str = diagnosis.status.value
                healing_action = diagnosis.healing_action

        # ── M5: Almacenar en memoria semantica ─────────────────────────
        # NOTA: El embedding se calcula a partir de las activaciones del
        # tercer forward pass (M2 final), mientras que el SCS proviene
        # de M1. Ambas metricas usan activaciones de forward passes
        # distintos. Esto es internamente consistente pero las dos
        # computaciones no comparten estado de activacion.
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

        elapsed_ms = (time.time() - start) * 1000

        return SymbolicResult(
            response=best_response,
            scs=best_scs,
            metrics=metrics,
            diagnosis=diagnosis_str,
            healing_action=healing_action,
            latency_ms=round(elapsed_ms, 1),
        )
