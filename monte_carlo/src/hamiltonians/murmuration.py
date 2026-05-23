"""4D Aerial Murmuration Hamiltonian for starling swarms under predator attack.

Extends IJRR23 (Molu et al. 2023) Eq. 46-47 to 4D aerial space.

State: x = (x₁, x₂, x₃, θ) ∈ R² × R × S¹
  - (x₁, x₂): horizontal position (m)
  - x₃: altitude (m)
  - θ: horizontal heading angle (rad)

Dynamics (extension of IJRR23 Eq. 12-14):
  Free agent: ẋ₁ = v·cos(θ), ẋ₂ = v·sin(θ), ẋ₃ = u_z_e, θ̇ = ⟨w_e⟩_r
  Attacked: ẋ₁ = -v_p + v_e·cos(θ) + ⟨w_e⟩_r·x₂, ẋ₂ = v_p·sin(θ) - ⟨w_e⟩_r·x₁,
           ẋ₃ = u_z_e - u_z_p, θ̇ = w_p - ⟨w_e⟩_r

Hamiltonian (min-max game, pursuer minimizes, evader maximizes):
  H_free = p₁·cos(θ) + p₂·sin(θ) + p₃·u_z_e + p₄·⟨w_e⟩_r
  H_att = p₁·(1-cos(θ)) - p₂·sin(θ) + γ·smooth_abs(p₃)
          - ω̄_p·smooth_abs(p₄) + ω̄_e·smooth_abs(p₂·x₁ - p₁·x₂ + p₄)
  H = min(H_free, H_att)
"""

import jax.numpy as jnp
from .base import Hamiltonian


class MurmuationHamiltonian4D(Hamiltonian):
    """4D aerial murmuration: state (x₁, x₂, x₃_alt, θ)."""

    def __init__(
        self,
        omega_e_bar: float = 1.0,
        omega_p_bar: float = 1.0,
        gamma_max: float = 0.5,
        n_neighbors: int = 7,
        neighbor_headings: list = None,
        smoothing_eps: float = 1e-4,
    ):
        """
        Parameters
        ----------
        omega_e_bar : float
            Angular speed bound for evader (flock).
        omega_p_bar : float
            Angular speed bound for pursuer (predator).
        gamma_max : float
            Climb/dive rate bound (m/s).
        n_neighbors : int
            Topological neighbor count (IJRR23 Eq. 13).
        neighbor_headings : list, optional
            Heading values for neighbors (for avg_heading computation).
        smoothing_eps : float
            Smoothing parameter for abs approximation.
        """
        self.omega_e_bar = omega_e_bar
        self.omega_p_bar = omega_p_bar
        self.gamma_max = gamma_max
        self.n_neighbors = n_neighbors
        self.neighbor_headings = neighbor_headings if neighbor_headings else []
        self.eps = smoothing_eps

    def _avg_heading(self, w_e: float) -> float:
        """Compute average heading (IJRR23 Eq. 13).

        ⟨w⟩_r = 1/(1+n_i) * (w_e + sum_j w_j)
        """
        if not self.neighbor_headings:
            return w_e
        n_i = len(self.neighbor_headings)
        return (w_e + sum(self.neighbor_headings)) / (1.0 + n_i)

    def __call__(self, t, x, p):
        """Evaluate Hamiltonian at (x, p).

        Parameters
        ----------
        t : float
            Time (unused, for compatibility).
        x : jnp.ndarray, shape (..., 4)
            State (x₁, x₂, x₃, θ).
        p : jnp.ndarray, shape (..., 4)
            Costate (p₁, p₂, p₃, p₄).

        Returns
        -------
        jnp.ndarray
            Hamiltonian value at (x, p).
        """
        p1, p2, p3, p4 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        x1, x2, x3, th = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        sa = lambda z: jnp.sqrt(z**2 + self.eps**2)

        # Flock average heading (constant per iteration)
        w_r = self._avg_heading(0.0)

        # Altitude game: evader maximizes, pursuer minimizes altitude rate
        # H_alt = γ_max · smooth_abs(p₃)
        H_alt = self.gamma_max * sa(p3)

        # Free-agent Hamiltonian (IJRR23 Eq. 46)
        H_free = p1 * jnp.cos(th) + p2 * jnp.sin(th) + H_alt + p4 * w_r

        # Attacked-agent Hamiltonian (IJRR23 Eq. 47 + altitude game)
        H_att = (
            p1 * (1.0 - jnp.cos(th))
            - p2 * jnp.sin(th)
            + H_alt
            - self.omega_p_bar * sa(p4)
            + self.omega_e_bar * sa(p2 * x1 - p1 * x2 + p4)
        )

        # Union operator = min (IJRR23 Eq. 16)
        return jnp.minimum(H_free, H_att)

    @property
    def state_dim(self) -> int:
        return 4


class MurmuationFlockHamiltonian4D(Hamiltonian):
    """Multi-flock 4D murmuration: applies min over n_f flocks."""

    def __init__(self, n_flocks: int = 1, per_flock_H=None, smoothing_eps: float = 1e-4):
        """
        Parameters
        ----------
        n_flocks : int
            Number of flocks.
        per_flock_H : list of Hamiltonian
            List of n_flocks Hamiltonian instances (one per flock).
        smoothing_eps : float
            Smoothing parameter.
        """
        self.n_flocks = n_flocks
        self.per_flock_H = per_flock_H or [
            MurmuationHamiltonian4D(smoothing_eps=smoothing_eps) for _ in range(n_flocks)
        ]
        self.eps = smoothing_eps

    def __call__(self, t, x, p):
        """Hamiltonian union across flocks (IJRR23 Eq. 21).

        For multi-flock safety, use min (union).
        """
        from jax import vmap

        vals = jnp.array([H(t, x, p) for H in self.per_flock_H])
        return jnp.min(vals, axis=0)

    @property
    def state_dim(self) -> int:
        return 4
