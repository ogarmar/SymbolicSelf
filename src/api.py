# src/api.py — REST API para SymbolicSelf
"""
FastAPI server que expone el pipeline SymbolicSelf como una API REST.

Endpoints:
  POST /refine    — Procesa pregunta + imagen → respuesta refinada con SCS
  GET  /health    — Estado del servidor y modelo cargado

Uso:
  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import time

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from PIL import Image

from src.config import MODEL_ID
from src.symbolic_self import SymbolicSelfPipeline

logger = logging.getLogger(__name__)

# ── Modelos de respuesta ───────────────────────────────────────────────


class RefineResponse(BaseModel):
    """Respuesta del endpoint /refine."""
    response: str
    scs: float
    diagnosis: str
    healing_action: str
    n_clusters: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""
    status: str
    model: str
    baseline_established: bool


# ── Aplicación FastAPI ─────────────────────────────────────────────────

app = FastAPI(
    title="SymbolicSelf API",
    description="Pipeline maestro M1 + M2 + M3 para VQA con coherencia simbólica.",
    version="1.0.0",
)

# Inicialización lazy del pipeline (al primer request)
_pipeline: SymbolicSelfPipeline | None = None


def get_pipeline() -> SymbolicSelfPipeline:
    """Inicialización lazy del pipeline (carga el modelo la primera vez)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Inicializando pipeline (primera llamada)...")
        _pipeline = SymbolicSelfPipeline()
    return _pipeline


# ── Endpoints ──────────────────────────────────────────────────────────


@app.post("/refine", response_model=RefineResponse)
async def refine(
    question: str = Form(..., description="Pregunta VQA"),
    image: UploadFile | None = File(None, description="Imagen (opcional)"),
    n_variants: int = Form(5, description="Nº variantes para self-polish"),
):
    """Procesa una pregunta + imagen y devuelve la respuesta refinada con SCS."""
    pipeline = get_pipeline()

    # Cargar imagen si se envió
    img = None
    if image is not None:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = pipeline.process(question, img, n_variants=n_variants)

    return RefineResponse(
        response=result.response,
        scs=result.scs,
        diagnosis=result.diagnosis,
        healing_action=result.healing_action,
        n_clusters=result.metrics.get("n_clusters", 0),
        latency_ms=result.latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Estado del servidor."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        model=MODEL_ID,
        baseline_established=pipeline._baseline_established,
    )
