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
_init_lock = asyncio.Lock()


async def _ensure_pipeline():
    """Inicializa pipeline con lock async (thread-safe, una sola vez)."""
    global _pipeline
    async with _init_lock:
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

    return RefineResponse(
        response=result.response,
        scs=result.scs,
        diagnosis=result.diagnosis,
        healing_action=result.healing_action,
        latency_ms=result.latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Estado del servidor (no carga el modelo si aun no esta listo)."""
    return HealthResponse(
        status="ok" if _pipeline is not None else "loading",
        model=MODEL_ID,
        baseline_established=_pipeline._baseline_established if _pipeline else False,
    )
