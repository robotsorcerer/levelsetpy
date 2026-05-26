"""4D Aerial murmuration dynamics (JAX implementation).

Implements IJRR23 Eq. 12-15 for 4D Dubins agents:
  x = (x₁, x₂, x₃, θ) ∈ R² × R × S¹

Absolute dynamics:
  ẋ₁ = v·cos(θ)
  ẋ₂ = v·sin(θ)
  ẋ₃ = u_z
  θ̇ = ⟨w⟩_r = (1/(1+n_i)) · (w + sum_j w_j)

Relative dynamics under predator attack:
  ẋ₁ = -v_p + v_e·cos(θ) + ⟨w_e⟩_r·x₂
  ẋ₂ = v_p·sin(θ) - ⟨w_e⟩_r·x₁
  ẋ₃ = u_z_e - u_z_p
  θ̇ = w_p - ⟨w_e⟩_r

Terminal cost (capture cylinder): g(x) = sqrt(x₁² + x₂²) - r_capture
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Tuple

import jax
import jax.numpy as jnp
from jax import vmap

if TYPE_CHECKING:
    from src.gpu_distribution import GPUDistributor


import sys 
import os
from os.path import join 
from pathlib import Path
try:
    _HERE = os.path.abspath(__file__)
except NameError:
    _HERE = Path.cwd()
_ROOT = os.path.dirname(os.path.dirname(_HERE)) 
sys.path.append(_ROOT)

from src.config import SolverConfig
from src.hj_sampler import HJReachabilitySampler
from src.hamiltonians.murmuration import MurmuationHamiltonian4D


def avg_heading_jax(w_e: float, neighbor_headings: jnp.ndarray) -> float:
    """Compute average heading (IJRR23 Eq. 13).

    ⟨w⟩_r = 1/(1+n_i) * (w_e + sum_j w_j)

    Parameters
    ----------
    w_e : float
        Agent's own heading control.
    neighbor_headings : jnp.ndarray, shape (n_neighbors,)
        Neighbors' heading values.

    Returns
    -------
    float
        Average heading ⟨w⟩_r.
    """
    n_i = neighbor_headings.shape[0]
    return (w_e + jnp.sum(neighbor_headings)) / (1.0 + n_i)


def abs_dynamics_4d(
    x: jnp.ndarray, w_r: float, u_z: float = 0.0, v: float = 1.0
) -> jnp.ndarray:
    """Absolute dynamics for single 4D agent (IJRR23 Eq. 12).

    Parameters
    ----------
    x : jnp.ndarray, shape (4,)
        State (x₁, x₂, x₃, θ).
    w_r : float
        Average heading rate ⟨w⟩_r.
    u_z : float
        Climb rate (m/s).
    v : float
        Linear speed (m/s).

    Returns
    -------
    jnp.ndarray, shape (4,)
        State derivative (ẋ₁, ẋ₂, ẋ₃, θ̇).
    """
    x1, x2, x3, theta = x[0], x[1], x[2], x[3]
    return jnp.array(
        [v * jnp.cos(theta), v * jnp.sin(theta), u_z, w_r]
    )


def rel_dynamics_4d(
    x: jnp.ndarray,
    w_r_e: float,
    u_z_e: float = 0.0,
    u_z_p: float = 0.0,
    v_p: float = 1.0,
    v_e: float = 1.0,
    w_p: float = 0.0,
) -> jnp.ndarray:
    """Relative dynamics under predator attack (IJRR23 Eq. 14).

    Parameters
    ----------
    x : jnp.ndarray, shape (4,)
        Relative state (x₁, x₂, x₃, θ).
    w_r_e : float
        Average heading rate of evader flock.
    u_z_e : float
        Evader climb rate (m/s).
    u_z_p : float
        Pursuer climb rate (m/s).
    v_p : float
        Pursuer linear speed (m/s).
    v_e : float
        Evader linear speed (m/s).
    w_p : float
        Pursuer angular speed (control input, rad/s).

    Returns
    -------
    jnp.ndarray, shape (4,)
        State derivative (ẋ₁, ẋ₂, ẋ₃, θ̇).
    """
    x1, x2, x3, theta = x[0], x[1], x[2], x[3]
    return jnp.array(
        [
            -v_p + v_e * jnp.cos(theta) + w_r_e * x2,
            v_p * jnp.sin(theta) - w_r_e * x1,
            u_z_e - u_z_p,
            w_p - w_r_e,
        ]
    )


def terminal_cost_4d(x: jnp.ndarray, r_capture: float = 0.2) -> float:
    """Capture cylinder cost (ignores altitude, IJRR23 Eq. 38).

    g(x) = sqrt(x₁² + x₂²) - r_capture

    Negative inside capture set, positive outside.

    Parameters
    ----------
    x : jnp.ndarray, shape (4,)
        State (x₁, x₂, x₃, θ).
    r_capture : float
        Capture radius (m).

    Returns
    -------
    float
        Cost value.
    """
    r_xy = jnp.linalg.norm(x[:2])
    return r_xy - r_capture


@dataclass
class FlockState:
    """State of a single flock of agents."""

    states: jnp.ndarray
    flock_id: int
    neighbor_graph: Optional[dict] = None

    @property
    def n_agents(self) -> int:
        return self.states.shape[0]


@dataclass
class PredatorState:
    """State of a single predator."""

    position: jnp.ndarray
    omega_max: float = 1.0
    gamma_max: float = 0.5
    speed: float = 1.0


class MurmuationSolverJAX4D:
    """HJ-Gauss solver for 4D aerial murmuration safety."""

    def __init__(
        self,
        cfg: SolverConfig,
        omega_e_bar: float = 1.0,
        omega_p_bar: float = 1.0,
        gamma_max: float = 0.5,
        distributor: Optional["GPUDistributor"] = None,
    ):
        """
        Parameters
        ----------
        cfg : SolverConfig
            Solver configuration.
        omega_e_bar : float
            Evader angular speed bound.
        omega_p_bar : float
            Pursuer angular speed bound.
        gamma_max : float
            Climb rate bound.
        distributor : GPUDistributor, optional
            Multi-GPU distributor for sharding across devices.
        """
        self.cfg = cfg
        self.omega_e_bar = omega_e_bar
        self.omega_p_bar = omega_p_bar
        self.gamma_max = gamma_max
        self.distributor = distributor

    def solve_single_flock(
        self, flock: FlockState, predator: PredatorState, t: float = 0.0
    ) -> Tuple[jnp.ndarray, float]:
        """Solve BRT for single flock under single predator.

        Parameters
        ----------
        flock : FlockState
            Flock state.
        predator : PredatorState
            Predator state.
        t : float
            Query time.

        Returns
        -------
        v : jnp.ndarray, shape (n_agents,)
            BRT values for each agent.
        wall_time : float
            Wall-clock solve time (seconds).
        """
        import time

        start = time.time()

        H = MurmuationHamiltonian4D(
            omega_e_bar=self.omega_e_bar,
            omega_p_bar=self.omega_p_bar,
            gamma_max=self.gamma_max,
        )
        solver = HJReachabilitySampler(H, terminal_cost_4d, self.cfg, self.distributor)
        v, _ = solver.solve_quasi_linear(flock.states, t=t)

        wall_time = time.time() - start
        return v, wall_time

    def solve_flock_system(
        self, flocks: List[FlockState], predators: List[PredatorState], t: float = 0.0
    ) -> Tuple[List[jnp.ndarray], float, float]:
        """Solve BRT for multi-flock, multi-predator system.

        Safety: min over all (flock, predator) pairs.

        Parameters
        ----------
        flocks : list of FlockState
            All flocks.
        predators : list of PredatorState
            All predators.
        t : float
            Query time.

        Returns
        -------
        safety_values : list of jnp.ndarray, length n_f
            BRT value for each agent in each flock; ``safety_values[f]`` has
            shape ``(n_agents_f,)`` and equals the per-agent minimum over all
            predators for flock ``f``.
        safe_fraction : float
            Fraction of all agents across all flocks with v > 0 (safe).
        wall_time : float
            Total wall-clock time.
        """
        import time

        start = time.time()

        n_f = len(flocks)
        n_p = len(predators)

        # Solve all (flock, predator) pairs.
        # safety_all_pairs[f] is a list of n_p arrays, each shape (n_agents_f,).
        safety_all_pairs: List[List[jnp.ndarray]] = []
        for flock in flocks:
            flock_safety: List[jnp.ndarray] = []
            for predator in predators:
                v, _ = self.solve_single_flock(flock, predator, t)
                flock_safety.append(v)
            safety_all_pairs.append(flock_safety)

        # Per-flock safety: min over all predators for that flock.
        # Result is a list of (n_agents_f,) arrays, one per flock.
        safety_values: List[jnp.ndarray] = [
            jnp.min(jnp.stack(flock_safety, axis=0), axis=0)
            for flock_safety in safety_all_pairs
        ]  # list of length n_f, each (n_agents_f,)

        # Global safe fraction: concatenate across all flocks, then threshold.
        all_v = jnp.concatenate(safety_values, axis=0)  # (total_agents,)
        safe_fraction = float(jnp.mean(all_v > 0))
        wall_time = time.time() - start

        return safety_values, safe_fraction, wall_time
