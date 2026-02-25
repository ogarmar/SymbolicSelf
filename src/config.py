# src/config.py — Configuración centralizada del proyecto SymbolicSelf
"""
Todas las rutas, hiperparámetros y constantes del proyecto viven aquí.
Nunca uses rutas absolutas hardcodeadas en otros ficheros.
"""

from pathlib import Path
import torch

# ── Rutas ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COCO_IMAGES_DIR = DATA_DIR / "vqa"  # val2017 images

# ── Modelo base ────────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
TORCH_DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Cuantización (RTX 4050 6GB) ────────────────────────────────────────────
# NOTA: llm_int8_enable_fp32_cpu_offload es SÓLO para modo int8 (LLM.int8()),
# NO debe usarse con load_in_4bit=True — provoca que bitsandbytes intente
# cuantizar tensores uint8 offloadeados desde CPU → ValueError en inferencia.
QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": TORCH_DTYPE,
    "bnb_4bit_use_double_quant": True,
}

# Limites de memoria para GPU 6GB
# IMPORTANTE: El modelo en 4-bit ocupa ~4.5GB. NO usar offload a CPU
# porque accelerate intenta re-cuantizar tensores uint8 → ValueError.
# Dejamos que todo el modelo viva en GPU con margen para KV-cache.
MAX_MEMORY = {
    0: "5.5GiB",
}

# ── Symbol Detector ────────────────────────────────────────────────────────
HOOK_LAYERS = [12, 15, 18]
PCA_COMPONENTS_TOKENS = 32   # Dimensiones PCA por token (antes: 64)
HDBSCAN_MIN_CLUSTER = 10     # Mínimo absoluto (dinámico: max(10, n_tokens//50))
HDBSCAN_MIN_SAMPLES = 5      # Anti-ruido (antes: 1)
MAX_TOKENS_PCA = 2000        # Limitar tokens para PCA por VRAM

# ── SCS (Symbolic Coherence Score) ─────────────────────────────────────────
SCS_ALPHA = 0.5   # Peso consistencia
SCS_BETA = 0.3    # Peso estabilidad
SCS_GAMMA = 0.2   # Peso alineación cross-modal

# ── Self-Polish ────────────────────────────────────────────────────────────
REFINE_TEMPLATES = [
    "Clarify this answer: {response}",
    "Make it more precise: {response}",
    "Complete missing details: {response}",
    "Verify the logic: {response}",
    "Be more specific: {response}",
]
DEFAULT_N_VARIANTS = 5
GENERATION_MAX_TOKENS = 50
GENERATION_TEMPERATURE = 0.7

# ── Self-Healing ───────────────────────────────────────────────────────────
ADVERSARIAL_STABILITY_THRESHOLD = 0.3
DRIFT_STABILITY_THRESHOLD = 0.75   # Antes 0.6 — subido para capturar drift borderline
ENTROPY_CHANGE_THRESHOLD = 0.5
