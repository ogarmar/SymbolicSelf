# src/model_loader.py — Carga centralizada del modelo LLaVA
"""
Singleton thread-safe para cargar LLaVA una sola vez.

Uso sync (tests, benchmark):
    from src.model_loader import load_model_sync
    model, processor = load_model_sync()

Uso async (FastAPI):
    from src.model_loader import get_model_async
    model, processor = await get_model_async()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from src.config import (
    LORA_ADAPTER_PATH,
    MAX_MEMORY,
    MODEL_ID,
    QUANTIZATION,
    TORCH_DTYPE,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ── Estado global (singleton) ──────────────────────────────────────────

_model: PreTrainedModel | None = None
_processor: LlavaNextProcessor | None = None
_async_lock = asyncio.Lock()


def _load_base_model(
    model_id: str = MODEL_ID,
    adapter_path: str | None = None,
) -> tuple[PreTrainedModel, LlavaNextProcessor]:
    """Carga pura (no singleton). Usada internamente."""
    logger.info("Cargando modelo %s ...", model_id)

    bnb_config = BitsAndBytesConfig(**QUANTIZATION)
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        max_memory=MAX_MEMORY,
        low_cpu_mem_usage=True,
    )

    # Configurar processor
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if hasattr(model.config, "vision_config"):
        processor.patch_size = getattr(model.config.vision_config, "patch_size", 14)
        processor.vision_feature_select_strategy = getattr(
            model.config, "vision_feature_select_strategy", "default"
        )

    # LoRA adapter
    effective_path = adapter_path or str(LORA_ADAPTER_PATH)
    if Path(effective_path).exists():
        from peft import PeftModel
        logger.info("Cargando LoRA adapter desde %s", effective_path)
        model = PeftModel.from_pretrained(model, effective_path)
        logger.info("LoRA adapter cargado.")

    logger.info("Modelo cargado.")
    return model, processor


def load_model_sync(
    model_id: str = MODEL_ID,
    adapter_path: str | None = None,
) -> tuple[PreTrainedModel, LlavaNextProcessor]:
    """Carga singleton sync (para tests y benchmark).

    Args:
        model_id: HuggingFace model ID.
        adapter_path: Ruta al adapter LoRA (None = usa LORA_ADAPTER_PATH si existe).

    Returns:
        (model, processor) reutilizados entre llamadas.
    """
    global _model, _processor
    if _model is None:
        _model, _processor = _load_base_model(model_id, adapter_path)
    return _model, _processor


async def get_model_async(
    model_id: str = MODEL_ID,
    adapter_path: str | None = None,
) -> tuple[PreTrainedModel, LlavaNextProcessor]:
    """Carga singleton async thread-safe (para FastAPI).

    Usa asyncio.Lock para evitar cargas duplicadas bajo concurrencia.
    """
    global _model, _processor
    async with _async_lock:
        if _model is None:
            _model, _processor = _load_base_model(model_id, adapter_path)
    return _model, _processor


def reset_singleton() -> None:
    """Limpia el singleton (para tests)."""
    global _model, _processor
    _model = None
    _processor = None
