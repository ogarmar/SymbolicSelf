# test/utils.py — Utilidades compartidas para tests
"""
Centraliza download_image y otras utilidades que estaban
duplicadas en multiples test files.
"""

from __future__ import annotations

import logging
from io import BytesIO

import requests
from PIL import Image

from src.config import DEFAULT_IMAGE_SIZE

logger = logging.getLogger(__name__)


def download_image(url: str) -> Image.Image | None:
    """Descarga una imagen desde una URL y la redimensiona.

    Usa DEFAULT_IMAGE_SIZE de config.py para limitar el numero de
    patches que LLaVA-Next AnyRes genera. Sin resize, imagenes COCO
    (640x480+) producen 2-3x mas tokens → OOM en GPUs de 6GB.

    Args:
        url: URL de la imagen.

    Returns:
        PIL.Image redimensionada, o None si falla la descarga.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.thumbnail(DEFAULT_IMAGE_SIZE)  # limit patches for 6GB GPU
        return image
    except Exception as e:
        logger.warning("Error descargando imagen %s: %s", url, e)
        return None
