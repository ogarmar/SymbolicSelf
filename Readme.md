# SymbolicSelf — TFG

Self-correcting symbolic reasoning pipeline for Vision-Language Models (LLaVA-v1.6-Mistral-7B).

## Architecture

| Module | File | Role |
|--------|------|------|
| **M1** Self-Polish | `src/m1_self_polish.py` | Generates N refined variants, selects best by SCS |
| **M2** Symbol Detector | `src/symbol_detector.py` | PyTorch hooks → PCA → HDBSCAN → emergent symbols + SCS |
| **M3** Self-Healing | `src/m3_self_healing.py` | Detects adversarial attacks & concept drift |
| **M4** Meta-Evolution | `src/m4_meta_evo.py` | Evolutionary optimisation of pipeline hyperparameters |
| **M5** Semantic Memory | `src/m5_semantic_memory.py` | In-memory vector store for retrieval-augmented context |
| **Pipeline** | `src/symbolic_self.py` | Orchestrates M1 → M5(retrieve) → M2 → M3 → M5(store) |
| **Model Loader** | `src/model_loader.py` | Thread-safe singleton for LLaVA loading |
| **API** | `src/api.py` | FastAPI REST server (`/refine`, `/health`) |
| **Config** | `src/config.py` | All paths, hyperparameters, and constants |
| **Utilities** | `src/symbol_utils.py` | Shared distribution functions (DRY) |

**M6** (Meta-Evolution at population level) is planned for a future milestone.

## Project Structure

```
TFGSymbolicSelf/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── model_loader.py        # Thread-safe singleton model loader
│   ├── symbol_detector.py     # M2: Hooks + PCA + HDBSCAN + SCS
│   ├── symbol_utils.py        # Shared distribution utilities
│   ├── m1_self_polish.py      # M1: Variant generation + selection
│   ├── m3_self_healing.py     # M3: Adversarial/drift detection
│   ├── m4_meta_evo.py         # M4: Evolutionary optimiser
│   ├── m5_semantic_memory.py  # M5: Semantic memory store
│   ├── symbolic_self.py       # Pipeline maestro
│   └── api.py                 # FastAPI REST server
├── test/
│   ├── utils.py               # Shared test utilities
│   ├── test_gpu_pipeline.py   # Full GPU integration test
│   ├── test_symbol_detector.py
│   ├── test_self_healing.py
│   ├── test_self_polish_scs.py
│   ├── test_symbolic_vqa.py   # End-to-end VQA with real dataset
│   ├── test_baseline_vqa2.py  # LLaVA baseline (no SymbolicSelf)
│   ├── check_gpu.py
│   └── prepare_vqa_subset.py
├── data/
│   ├── vqa/                   # COCO val2017 images
│   └── vqa_loader.py          # VQA dataset loader
├── docs/
├── download_models.py         # Download & cache models
└── requirements.txt
```

## Quick Start

```bash
# 1. Create environment
conda create -n symbolic_self python=3.11 -y
conda activate symbolic_self

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (first time only)
python download_models.py

# 4. Verify GPU
python test/check_gpu.py

# 5. Run baseline test
python test/test_baseline_vqa2.py

# 6. Run Symbol Detector test
python test/test_symbol_detector.py

# 7. Run full pipeline test
python test/test_self_polish_scs.py
```

## Requirements

- **GPU**: NVIDIA with ≥6GB VRAM (tested on RTX 4050)
- **RAM**: ≥16GB (CPU offload for model layers)
- **Python**: 3.10+
- **CUDA**: 11.8+

## SCS (Symbolic Coherence Score)

```
SCS = α · consistency + β · stability + γ · cross_modal
```

Where:
- **Consistency**: Jaccard similarity between current and baseline symbol clusters
- **Stability**: Jaccard similarity between current and previous extraction (temporal)
- **Cross-modal**: Alignment between visual encoder and language decoder activations
