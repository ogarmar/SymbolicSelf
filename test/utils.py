# test/utils.py — Utilidades compartidas para tests
"""
FIX 5: Centraliza download_image y otras utilidades que estaban
duplicadas en multiples test files.
"""

from __future__ import annotations

import logging
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def download_image(url: str) -> Image.Image | None:
    """Descarga una imagen desde una URL.

    Args:
        url: URL de la imagen.

    Returns:
        PIL.Image o None si falla la descarga.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.warning("Error descargando imagen %s: %s", url, e)
        return None
