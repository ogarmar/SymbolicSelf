# src/api.py — REST API para SymbolicSelf
"""
FastAPI server que expone el pipeline SymbolicSelf como una API REST.

Endpoints:
  POST /refine    — Procesa pregunta + imagen -> respuesta refinada con SCS
  GET  /health    — Estado del servidor y modelo cargado

Uso:
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1

IMPORTANTE: usar siempre --workers 1 (modelo ocupa ~4.5GB de 6GB VRAM).
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from PIL import Image

from src.config import MODEL_ID, LORA_ADAPTER_PATH

logger = logging.getLogger(__name__)


# ── Modelos de respuesta ───────────────────────────────────────────────

class RefineResponse(BaseModel):
    """Respuesta del endpoint /refine."""
    response: str
    scs: float
    # FIX 6: Campos de metricas individuales
    consistency: float
    stability: float
    cross_modal: float
    diagnosis: str
    healing_action: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""
    status: str
    model: str
    baseline_established: bool


# ── Estado global ──────────────────────────────────────────────────────

_pipeline = None

# FIX 5: Lazy Lock — no crear asyncio.Lock al importar (rompe tests).
# Se instancia la primera vez que se necesita.
_init_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Lazy Lock pattern: crea el lock solo cuando se necesita."""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


async def _ensure_pipeline():
    """Inicializa pipeline con lock async (thread-safe, una sola vez)."""
    global _pipeline
    lock = _get_lock()
    async with lock:
        if _pipeline is None:
            from src.symbolic_self import SymbolicSelfPipeline
            logger.info("Inicializando pipeline con adapter...")
            _pipeline = SymbolicSelfPipeline(
                adapter_path=str(LORA_ADAPTER_PATH),
            )
    return _pipeline


# ── Lifespan: carga modelo al arrancar ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar — no en la primera request."""
    await _ensure_pipeline()
    logger.info("Pipeline cargado y listo.")
    yield
    logger.info("Servidor apagado.")


# ── Aplicacion FastAPI ─────────────────────────────────────────────────

app = FastAPI(
    title="SymbolicSelf API",
    description="Pipeline M1 + M2 + M3 para VQA con coherencia simbolica.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/refine", response_model=RefineResponse)
async def refine(
    question: str = Form(..., description="Pregunta VQA"),
    image: UploadFile | None = File(None, description="Imagen (opcional)"),
    n_variants: int = Form(5, description="N variantes para self-polish"),
):
    """Procesa pregunta + imagen -> respuesta refinada con SCS.

    Ejecuta la inferencia en un thread separado para no bloquear
    el event loop (el health check sigue respondiendo).
    """
    pipeline = await _ensure_pipeline()

    # Cargar imagen si se envio
    img = None
    if image is not None:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Ejecutar inference fuera del event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.process(question, img, n_variants=n_variants),
    )

    # FIX 6: Mapear metricas individuales desde result.metrics
    return RefineResponse(
        response=result.response,
        scs=result.scs,
        consistency=result.metrics.consistency,
        stability=result.metrics.stability,
        cross_modal=result.metrics.cross_modal,
        diagnosis=result.diagnosis,
        healing_action=result.healing_action,
        latency_ms=result.latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    status = _pipeline.get_status() if _pipeline else {}
    return HealthResponse(
        status="ok" if _pipeline is not None else "loading",
        model=MODEL_ID,
        # CPython GIL makes this bool read safe without a lock;
        # revisit if moving to sub-interpreters.
        baseline_established=status.get("baseline_established", False),
    )
