# src/m5_semantic_memory.py — MÓDULO 5: Semantic Memory
"""
Memoria semántica para el pipeline SymbolicSelf.

Almacena embeddings de respuestas pasadas + contexto simbólico
para mejorar la coherencia a largo plazo y permitir retrieval
de patrones similares.

Implementación:
  - Vector store en memoria (NumPy) con cosine similarity
  - Cada entrada: (question, answer, symbols, scs, timestamp)
  - Retrieval: buscar los K vecinos más cercanos por embedding

NOTA: Implementación básica in-memory. Para producción usar FAISS o ChromaDB.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Una entrada en la memoria semántica."""
    question: str
    answer: str
    embedding: np.ndarray         # Embedding del par (question, answer)
    symbols: np.ndarray           # Cluster IDs de extract_symbols
    scs: float                    # SCS score cuando se generó
    timestamp: float = 0.0        # Epoch timestamp
    metadata: dict = field(default_factory=dict)


class SemanticMemory:
    """Vector store in-memory para memoria semántica del pipeline.

    Almacena pares (pregunta, respuesta) con sus embeddings y símbolos.
    Permite retrieval por similitud para contextualizar futuras respuestas.
    """

    def __init__(self, max_entries: int = 1000, embedding_dim: int = 4096) -> None:
        self.max_entries = max_entries
        self.embedding_dim = embedding_dim
        # deque con maxlen hace evicción FIFO automatica en O(1)
        self.entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._embeddings_matrix: np.ndarray | None = None  # Cache para búsqueda rápida

    def _invalidate_cache(self) -> None:
        """Invalida la matriz de embeddings cacheada."""
        self._embeddings_matrix = None

    def _build_cache(self) -> None:
        """Construye la matriz de embeddings para búsqueda vectorial."""
        if not self.entries:
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.stack([e.embedding for e in self.entries])

    # ── Almacenamiento ─────────────────────────────────────────────────

    def store(
        self,
        question: str,
        answer: str,
        embedding: np.ndarray,
        symbols: np.ndarray,
        scs: float,
        metadata: dict | None = None,
    ) -> None:
        """Almacena una nueva entrada en la memoria.

        La deque con maxlen evicta automaticamente la entrada más antigua.
        """
        entry = MemoryEntry(
            question=question,
            answer=answer,
            embedding=embedding,
            symbols=symbols,
            scs=scs,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        was_full = len(self.entries) >= self.max_entries
        self.entries.append(entry)  # O(1) — deque evicta la más antigua si llena
        if was_full:
            logger.debug("Memoria llena — evicción automática de entrada más antigua.")

        self._invalidate_cache()
        logger.debug("Memoria: %d/%d entradas.", len(self.entries), self.max_entries)

    # ── Recuperación ───────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """Busca las K entradas más similares por cosine similarity.

        Args:
            query_embedding: Embedding de la consulta.
            top_k: Número máximo de resultados.
            min_similarity: Similitud mínima para incluir.

        Returns:
            Lista de (entry, similarity_score) ordenada por similitud desc.
        """
        if not self.entries:
            return []

        if self._embeddings_matrix is None:
            self._build_cache()

        # Cosine similarity vectorizada
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        matrix_norms = self._embeddings_matrix / (
            np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )
        similarities = matrix_norms @ query_norm

        # Top-K
        if top_k < len(similarities):
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= min_similarity:
                results.append((self.entries[idx], sim))

        return results

    # ── Utilidades ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Estadísticas de la memoria."""
        if not self.entries:
            return {"n_entries": 0, "avg_scs": 0.0, "age_seconds": 0.0}

        now = time.time()
        return {
            "n_entries": len(self.entries),
            "avg_scs": float(np.mean([e.scs for e in self.entries])),
            "oldest_age_seconds": now - self.entries[0].timestamp,
            "newest_age_seconds": now - self.entries[-1].timestamp,
        }

    def clear(self) -> None:
        """Vacía la memoria."""
        self.entries.clear()
        self._invalidate_cache()
        logger.info("Memoria semántica vaciada.")

    def __len__(self) -> int:
        return len(self.entries)
