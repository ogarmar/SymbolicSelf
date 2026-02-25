# src/m4_meta_evo.py — MÓDULO 4: Meta-Evolutionary Optimizer
"""
Optimización evolutiva de los hiperparámetros del pipeline SymbolicSelf
usando Differential Evolution.

Optimiza:
  - Pesos SCS (α, β, γ)
  - N variantes para M1
  - Temperatura de generación
  - Parámetros HDBSCAN (min_cluster, min_samples)

Estrategia:
  1. Genoma = vector de hiperparámetros
  2. Fitness = accuracy VQA sobre un mini-batch
  3. Differential Evolution para buscar óptimos

NOTA: Implementación funcional pero básica. Sin paralelismo GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyGenome:
    """Genoma de una estrategia de self-correction."""
    scs_alpha: float = 0.5
    scs_beta: float = 0.3
    scs_gamma: float = 0.2
    n_variants: int = 5
    temperature: float = 0.7
    hdbscan_min_cluster: int = 10
    hdbscan_min_samples: int = 5

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.scs_alpha, self.scs_beta, self.scs_gamma,
            self.n_variants, self.temperature,
            self.hdbscan_min_cluster, self.hdbscan_min_samples,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "StrategyGenome":
        # Clamp y normalizar pesos SCS para que sumen 1
        alpha, beta, gamma = max(0, v[0]), max(0, v[1]), max(0, v[2])
        total = alpha + beta + gamma
        if total > 0:
            alpha, beta, gamma = alpha / total, beta / total, gamma / total
        else:
            alpha, beta, gamma = 0.33, 0.33, 0.34

        return cls(
            scs_alpha=alpha,
            scs_beta=beta,
            scs_gamma=gamma,
            n_variants=max(1, int(round(v[3]))),
            temperature=max(0.1, min(2.0, v[4])),
            hdbscan_min_cluster=max(3, int(round(v[5]))),
            hdbscan_min_samples=max(1, int(round(v[6]))),
        )


@dataclass
class EvolutionResult:
    """Resultado de la optimización evolutiva."""
    best_genome: StrategyGenome
    best_fitness: float
    generations: int
    history: list[float] = field(default_factory=list)


class MetaEvolutionaryOptimizer:
    """Optimizador evolutivo de hiperparámetros del pipeline.

    Usa Differential Evolution para encontrar la mejor combinación
    de parámetros que maximice VQA accuracy.
    """

    def __init__(
        self,
        population_size: int = 10,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.7,
        max_generations: int = 20,
        seed: int = 42,
    ) -> None:
        self.pop_size = population_size
        self.F = mutation_factor
        self.CR = crossover_rate
        self.max_gen = max_generations
        self.rng = np.random.default_rng(seed)

        # Límites de cada dimensión del genoma
        self.bounds = np.array([
            [0.0, 1.0],    # scs_alpha
            [0.0, 1.0],    # scs_beta
            [0.0, 1.0],    # scs_gamma
            [1.0, 10.0],   # n_variants
            [0.1, 2.0],    # temperature
            [3.0, 50.0],   # hdbscan_min_cluster
            [1.0, 20.0],   # hdbscan_min_samples
        ])

    def _init_population(self) -> np.ndarray:
        """Genera población inicial aleatoria dentro de los bounds."""
        n_dims = self.bounds.shape[0]
        pop = self.rng.random((self.pop_size, n_dims))
        for d in range(n_dims):
            pop[:, d] = self.bounds[d, 0] + pop[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        return pop

    def _mutate(self, pop: np.ndarray, idx: int) -> np.ndarray:
        """Mutación DE/rand/1."""
        candidates = [i for i in range(self.pop_size) if i != idx]
        a, b, c = self.rng.choice(candidates, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        # Clamp a bounds
        for d in range(len(mutant)):
            mutant[d] = np.clip(mutant[d], self.bounds[d, 0], self.bounds[d, 1])
        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Crossover binomial."""
        n_dims = len(target)
        trial = target.copy()
        j_rand = self.rng.integers(n_dims)
        for j in range(n_dims):
            if self.rng.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def optimize(self, fitness_fn) -> EvolutionResult:
        """Ejecuta la optimización evolutiva.

        Args:
            fitness_fn: Callable que recibe un StrategyGenome y retorna
                       un float (accuracy / fitness). Se maximiza.

        Returns:
            EvolutionResult con el mejor genoma encontrado.
        """
        logger.info("🧬 Iniciando Meta-Evolutionary Optimization (%d gen, pop=%d)",
                     self.max_gen, self.pop_size)

        pop = self._init_population()
        fitness = np.zeros(self.pop_size)

        # Evaluar población inicial
        for i in range(self.pop_size):
            genome = StrategyGenome.from_vector(pop[i])
            fitness[i] = fitness_fn(genome)
            logger.info("  Init %d/%d: fitness=%.4f", i + 1, self.pop_size, fitness[i])

        history = [float(fitness.max())]

        # Evolucionar
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                mutant = self._mutate(pop, i)
                trial_vec = self._crossover(pop[i], mutant)
                trial_genome = StrategyGenome.from_vector(trial_vec)
                trial_fitness = fitness_fn(trial_genome)

                if trial_fitness >= fitness[i]:
                    pop[i] = trial_vec
                    fitness[i] = trial_fitness

            best_idx = np.argmax(fitness)
            best_fit = fitness[best_idx]
            history.append(best_fit)

            logger.info("  Gen %d/%d: best=%.4f, mean=%.4f",
                         gen + 1, self.max_gen, best_fit, fitness.mean())

        best_idx = np.argmax(fitness)
        best_genome = StrategyGenome.from_vector(pop[best_idx])

        logger.info("🏆 Mejor genoma: %s (fitness=%.4f)", best_genome, fitness[best_idx])

        return EvolutionResult(
            best_genome=best_genome,
            best_fitness=float(fitness[best_idx]),
            generations=self.max_gen,
            history=history,
        )
