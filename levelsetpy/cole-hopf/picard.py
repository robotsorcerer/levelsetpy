"""Depth-first nonlinear Monte Carlo Picard evaluation for transformed HJ PDEs.

The evaluator computes one or more point queries without a state-space grid.  It
uses the Gaussian mild representation for both the transformed value and its
gradient.  Source-time importance sampling removes the terminal-time score
singularity.  Recursion is depth first, so live coordinate storage grows with
Picard depth, sample count, and state dimension rather than a tensor grid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
InitialValue = Callable[[Array], Array]
Coefficient = Callable[[float, Array], tuple[float, float, Array, float]]
Hamiltonian = Callable[[float, Array, Array], float]


@dataclass
class EvaluationStats:
    """Operation counts that accompany one point evaluation."""

    heat_evaluations: int = 0
    source_evaluations: int = 0
    gaussian_vectors: int = 0
    maximum_recursion_depth: int = 0


@dataclass(frozen=True)
class PicardResult:
    """Recovered value, transformed variables, and reproducibility metadata."""

    value: float
    gradient: Array
    transformed_value: float
    transformed_gradient: Array
    level: int
    samples: int
    seed: int
    stats: dict[str, int]


class NestedPicardSolver:
    """Evaluate a nonlinear transformed HJ Picard iterate at a point.

    The transformed equation is

        z_t - (delta/2) Delta z = F(t, x, z, Dz),

    with ``z(0,x)=exp(-c(0,x) g(x))``.  Level zero is homogeneous heat
    flow.  Each higher level freezes the source at the preceding level and
    evaluates the resulting Duhamel formula by nested Monte Carlo sampling.

    This algorithm favors bounded live memory over runtime.  A level ``k``
    query requires a tree of preceding-level queries and therefore has a
    rapidly growing operation count.  The scaling report records that cost.
    """

    def __init__(
        self,
        initial_value: InitialValue,
        coefficient: Coefficient,
        hamiltonian: Hamiltonian,
        *,
        delta: float,
        samples: int,
    ) -> None:
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        if int(samples) != samples or samples < 2 or samples % 2:
            raise ValueError("samples must be a positive even integer")
        self.initial_value = initial_value
        self.coefficient = coefficient
        self.hamiltonian = hamiltonian
        self.delta = float(delta)
        self.samples = int(samples)

    def evaluate(self, level: int, time: float, state: Array, *, seed: int) -> PicardResult:
        """Return one Picard-level point estimate and recovered HJ value."""
        state = np.asarray(state, dtype=float)
        if state.ndim != 1 or state.size < 1 or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite one-dimensional array")
        if int(level) != level or level < 0:
            raise ValueError("level must be a nonnegative integer")
        if not np.isfinite(time) or time <= 0:
            raise ValueError("time must be finite and positive")
        rng = np.random.default_rng(seed)
        stats = EvaluationStats()
        z, dz = self._iterate(int(level), float(time), state, rng, stats, 0)
        c, _, dc, _ = self._coefficient(float(time), state)
        if not np.isfinite(z) or z <= 0:
            raise FloatingPointError("Picard estimate is nonpositive or nonfinite")
        value = -np.log(z) / c
        gradient = -dz / (c * z) + np.log(z) * dc / c**2
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("logarithmic recovery is nonfinite")
        return PicardResult(
            value=float(value),
            gradient=np.asarray(gradient, dtype=float),
            transformed_value=float(z),
            transformed_gradient=np.asarray(dz, dtype=float),
            level=int(level),
            samples=self.samples,
            seed=int(seed),
            stats=asdict(stats),
        )

    def _coefficient(self, time: float, state: Array) -> tuple[float, float, Array, float]:
        c, ct, dc, lapc = self.coefficient(time, state)
        c = float(c)
        ct = float(ct)
        lapc = float(lapc)
        dc = np.asarray(dc, dtype=float)
        if dc.shape != state.shape:
            raise ValueError("coefficient gradient must match the state shape")
        if c <= 0 or not np.isfinite(c) or not np.isfinite(ct + lapc) or not np.all(np.isfinite(dc)):
            raise ValueError("coefficient and its derivatives must be finite, with c positive")
        return c, ct, dc, lapc

    def _initial_transform(self, states: Array) -> Array:
        states = np.asarray(states, dtype=float)
        values = np.asarray(self.initial_value(states), dtype=float)
        expected = states.shape[:-1]
        if values.shape != expected:
            raise ValueError(f"initial_value must return shape {expected}, received {values.shape}")
        flattened = states.reshape(-1, states.shape[-1])
        c0 = np.fromiter(
            (self._coefficient(0.0, point)[0] for point in flattened),
            dtype=float,
            count=len(flattened),
        ).reshape(expected)
        transformed = np.exp(-c0 * values)
        if not np.all(np.isfinite(transformed)) or np.any(transformed <= 0):
            raise FloatingPointError("initial transformed values must be finite and positive")
        return transformed

    def _heat(self, time: float, state: Array, rng: np.random.Generator,
              stats: EvaluationStats) -> tuple[float, Array]:
        dimension = state.size
        pairs = self.samples // 2
        noise = rng.standard_normal((pairs, dimension))
        scale = np.sqrt(self.delta * time)
        plus = self._initial_transform(state + scale * noise)
        minus = self._initial_transform(state - scale * noise)
        stats.heat_evaluations += 1
        stats.gaussian_vectors += self.samples
        value = float(np.mean(0.5 * (plus + minus)))
        gradient = np.mean(noise * (plus - minus)[:, None], axis=0) / (2.0 * scale)
        return value, gradient

    def _source(self, time: float, state: Array, z: float, dz: Array) -> float:
        if not np.isfinite(z) or z <= 0:
            raise FloatingPointError("preceding Picard estimate lost positivity")
        c, ct, dc, lapc = self._coefficient(time, state)
        value = -np.log(z) / c
        gradient = -dz / (c * z) + np.log(z) * dc / c**2
        h = float(self.hamiltonian(time, state, gradient))
        algebraic = c * (h - 0.5 * self.delta * c * float(np.dot(gradient, gradient)))
        derivative = (
            -ct * value
            + 0.5 * self.delta * lapc * value
            - 0.5 * self.delta * float(np.dot(dc, dc)) * value**2
            + self.delta * (1.0 - c * value) * float(np.dot(dc, gradient))
        )
        source = z * (algebraic + derivative)
        if not np.isfinite(source):
            raise FloatingPointError("transformed source is nonfinite")
        return float(source)

    def _iterate(
        self,
        level: int,
        time: float,
        state: Array,
        rng: np.random.Generator,
        stats: EvaluationStats,
        depth: int,
    ) -> tuple[float, Array]:
        stats.maximum_recursion_depth = max(stats.maximum_recursion_depth, depth)
        heat_value, heat_gradient = self._heat(time, state, rng, stats)
        if level == 0:
            return heat_value, heat_gradient

        # S=t(1-U^2) has density [2 sqrt(t(t-s))]^{-1}.  Its reciprocal
        # cancels the score singularity in the source-gradient estimator.
        pairs = self.samples // 2
        u = 1.0 - rng.random(pairs)
        lag = time * u**2
        source_time = time - lag
        noise = rng.standard_normal((pairs, state.size))
        displacement = np.sqrt(self.delta * lag)[:, None] * noise
        shifted_plus = state + displacement
        shifted_minus = state - displacement
        stats.gaussian_vectors += self.samples

        source_plus = np.empty(pairs, dtype=float)
        source_minus = np.empty(pairs, dtype=float)
        for index in range(pairs):
            previous_value, previous_gradient = self._iterate(
                level - 1,
                float(source_time[index]),
                shifted_plus[index],
                rng,
                stats,
                depth + 1,
            )
            source_plus[index] = self._source(
                float(source_time[index]), shifted_plus[index], previous_value, previous_gradient
            )
            previous_value, previous_gradient = self._iterate(
                level - 1,
                float(source_time[index]),
                shifted_minus[index],
                rng,
                stats,
                depth + 1,
            )
            source_minus[index] = self._source(
                float(source_time[index]), shifted_minus[index], previous_value, previous_gradient
            )
        stats.source_evaluations += self.samples

        value = heat_value + float(np.mean(time * u * (source_plus + source_minus)))
        gradient = heat_gradient + np.mean(
            np.sqrt(time / self.delta) * noise * (source_plus - source_minus)[:, None], axis=0
        )
        return value, gradient


def constant_coefficient(value: float) -> Coefficient:
    """Return a positive constant transformation coefficient callback."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError("coefficient value must be finite and positive")

    def coefficient(time: float, state: Array) -> tuple[float, float, Array, float]:
        del time
        return float(value), 0.0, np.zeros_like(state, dtype=float), 0.0

    return coefficient
