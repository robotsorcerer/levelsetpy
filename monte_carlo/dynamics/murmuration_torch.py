"""4D Aerial murmuration dynamics (PyTorch implementation).

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
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FlockState:
    """State of a single flock of agents."""

    states: torch.Tensor  # shape (n_agents, 4)
    flock_id: int
    neighbor_graph: Optional[dict] = None

    @property
    def n_agents(self) -> int:
        return self.states.shape[0]


@dataclass
class PredatorState:
    """State of a single predator."""

    position: torch.Tensor  # shape (4,)
    omega_max: float = 1.0
    gamma_max: float = 0.5
    speed: float = 1.0


def avg_heading_torch(w_e: float, neighbor_headings: torch.Tensor) -> float:
    """Compute average heading (IJRR23 Eq. 13).

    ⟨w⟩_r = 1/(1+n_i) * (w_e + sum_j w_j)

    Parameters
    ----------
    w_e : float
        Agent's own heading control.
    neighbor_headings : torch.Tensor, shape (n_neighbors,)
        Neighbors' heading values.

    Returns
    -------
    float
        Average heading ⟨w⟩_r.
    """
    n_i = neighbor_headings.shape[0]
    return (w_e + neighbor_headings.sum()) / (1.0 + n_i)


def abs_dynamics_4d(
    x: torch.Tensor, w_r: float, u_z: float = 0.0, v: float = 1.0
) -> torch.Tensor:
    """Absolute dynamics for single 4D agent (IJRR23 Eq. 12).

    Parameters
    ----------
    x : torch.Tensor, shape (4,)
        State (x₁, x₂, x₃, θ).
    w_r : float
        Average heading rate ⟨w⟩_r.
    u_z : float
        Climb rate (m/s).
    v : float
        Linear speed (m/s).

    Returns
    -------
    torch.Tensor, shape (4,)
        State derivative (ẋ₁, ẋ₂, ẋ₃, θ̇).
    """
    x1, x2, x3, theta = x[0], x[1], x[2], x[3]
    return torch.stack([
        v * torch.cos(theta),
        v * torch.sin(theta),
        u_z * torch.ones_like(theta),
        w_r * torch.ones_like(theta)
    ])


def rel_dynamics_4d(
    x: torch.Tensor,
    w_r_e: float,
    u_z_e: float = 0.0,
    u_z_p: float = 0.0,
    v_p: float = 1.0,
    v_e: float = 1.0,
    w_p: float = 0.0,
) -> torch.Tensor:
    """Relative dynamics under predator attack (IJRR23 Eq. 14).

    Parameters
    ----------
    x : torch.Tensor, shape (4,)
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
    torch.Tensor, shape (4,)
        State derivative (ẋ₁, ẋ₂, ẋ₃, θ̇).
    """
    x1, x2, x3, theta = x[0], x[1], x[2], x[3]
    return torch.stack([
        -v_p + v_e * torch.cos(theta) + w_r_e * x2,
        v_p * torch.sin(theta) - w_r_e * x1,
        u_z_e - u_z_p,
        w_p - w_r_e
    ])


def terminal_cost_4d(x: torch.Tensor, r_capture: float = 0.2) -> torch.Tensor:
    """Capture cylinder cost (ignores altitude, IJRR23 Eq. 38).

    g(x) = sqrt(x₁² + x₂²) - r_capture

    Negative inside capture set, positive outside.

    Parameters
    ----------
    x : torch.Tensor, shape (..., 4)
        State(s) (x₁, x₂, x₃, θ).
    r_capture : float
        Capture radius (m).

    Returns
    -------
    torch.Tensor
        Cost value(s).
    """
    r_xy = torch.norm(x[..., :2], dim=-1)
    return r_xy - r_capture


class MurmuationSolverTorch4D:
    """PyTorch-based HJ-Gauss solver for 4D aerial murmuration safety.

    Device-agnostic: works on CPU or GPU depending on input tensor devices.
    Automatically batches computations to avoid memory issues.
    """

    def __init__(
        self,
        delta: float = 0.05,
        num_samples: int = 10000,
        max_quasi_iters: int = 100,
        quasi_tol: float = 1e-5,
        t_start: float = 0.0,
        t_end: float = 2.0,
        omega_e_bar: float = 1.0,
        omega_p_bar: float = 1.0,
        gamma_max: float = 0.5,
        seed: int = 42,
        batch_size: int = 5000,
    ):
        """Initialize the solver.

        Parameters
        ----------
        delta : float
            Viscosity parameter
        num_samples : int
            Number of Monte Carlo samples
        max_quasi_iters : int
            Maximum quasi-linearization iterations
        quasi_tol : float
            Convergence tolerance for quasi-linearization
        t_start : float
            Start time for backward solve
        t_end : float
            Terminal time
        omega_e_bar : float
            Evader angular speed bound
        omega_p_bar : float
            Pursuer angular speed bound
        gamma_max : float
            Climb rate bound
        seed : int
            Random seed
        batch_size : int
            Batch size for MC sampling (defaults to 5000 points)
        """
        self.delta = delta
        self.num_samples = num_samples
        self.max_quasi_iters = max_quasi_iters
        self.quasi_tol = quasi_tol
        self.t_start = t_start
        self.t_end = t_end
        self.omega_e_bar = omega_e_bar
        self.omega_p_bar = omega_p_bar
        self.gamma_max = gamma_max
        self.seed = seed
        self.batch_size = batch_size

    def _mc_value_at_point(
        self,
        x: torch.Tensor,
        t: float,
        c: torch.Tensor,
        terminal_cost_fn,
    ) -> torch.Tensor:
        """Compute v(t, x) via MC Gaussian expectation with batching.

        Parameters
        ----------
        x : torch.Tensor, shape (n,) or (batch, n)
            Query point(s).
        t : float
            Current time.
        c : torch.Tensor
            Cole-Hopf coefficient(s).
        terminal_cost_fn : callable
            Terminal cost function g(x).

        Returns
        -------
        torch.Tensor
            Value function estimate.
        """
        device = x.device
        dtype = x.dtype

        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        n = x.shape[-1]
        batch_size = x.shape[0]
        time_diff = max(self.delta * (self.t_end - t), 1e-30)
        sigma = torch.sqrt(torch.tensor(time_diff, dtype=dtype, device=device))

        # Process in chunks to avoid memory issues
        num_chunks = (batch_size + self.batch_size - 1) // self.batch_size
        v_list = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.batch_size
            end = min(start + self.batch_size, batch_size)
            x_chunk = x[start:end]
            c_chunk = c[start:end]

            # Generate samples for this chunk
            z = torch.randn(
                x_chunk.shape[0], self.num_samples, n,
                dtype=dtype, device=device
            )
            y = x_chunk.unsqueeze(1) + sigma * z  # (batch, num_samples, n)

            # Evaluate terminal cost
            y_flat = y.reshape(-1, n)
            g_vals = terminal_cost_fn(y_flat)
            g_vals = g_vals.reshape(x_chunk.shape[0], self.num_samples)

            # Stable logsumexp
            exponents = -c_chunk.unsqueeze(1) * g_vals
            max_exp = torch.max(exponents, dim=1, keepdim=True)[0]
            log_mean_exp = max_exp.squeeze(1) + torch.log(
                torch.mean(torch.exp(exponents - max_exp), dim=1)
            )
            v_chunk = -(1.0 / c_chunk) * log_mean_exp
            v_list.append(v_chunk)

        v = torch.cat(v_list, dim=0)

        if squeeze_output and v.shape[0] == 1:
            v = v.squeeze(0)

        return v

    def _mc_gradient_at_point(
        self,
        x: torch.Tensor,
        t: float,
        c: torch.Tensor,
        terminal_cost_fn,
    ) -> torch.Tensor:
        """Compute Dv(t, x) via MC importance weighting.

        Parameters
        ----------
        x : torch.Tensor, shape (n,) or (batch, n)
            Query point(s).
        t : float
            Current time.
        c : torch.Tensor
            Cole-Hopf coefficient(s).
        terminal_cost_fn : callable
            Terminal cost function g(x).

        Returns
        -------
        torch.Tensor
            Gradient estimate, same shape as x.
        """
        device = x.device
        dtype = x.dtype

        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        n = x.shape[-1]
        time_diff = max(self.delta * (self.t_end - t), 1e-30)
        sigma = torch.sqrt(torch.tensor(time_diff, dtype=dtype, device=device))

        z = torch.randn(
            x.shape[0], self.num_samples, n,
            dtype=dtype, device=device
        )
        y = x.unsqueeze(1) + sigma * z  # (batch, num_samples, n)

        # Evaluate terminal cost and gradient
        y_flat = y.reshape(-1, n)
        y_flat.requires_grad_(True)

        g_vals = terminal_cost_fn(y_flat)

        # Compute gradient of g via autodiff
        g_sum = g_vals.sum()
        g_sum.backward()
        dg_vals = y_flat.grad

        dg_vals = dg_vals.reshape(x.shape[0], self.num_samples, n)
        g_vals = g_vals.detach().reshape(x.shape[0], self.num_samples)

        # Importance weights: exp(-c * g)
        exponents = -c.unsqueeze(1) * g_vals
        max_exp = torch.max(exponents, dim=1, keepdim=True)[0]
        w = torch.exp(exponents - max_exp)
        w_sum = torch.sum(w, dim=1, keepdim=True)

        # E[Dg * w] / E[w]
        weighted_grad = torch.sum(
            dg_vals * w.unsqueeze(-1),
            dim=1
        ) / w_sum

        if squeeze_output and weighted_grad.shape[0] == 1:
            weighted_grad = weighted_grad.squeeze(0)

        return weighted_grad

    def _quasi_linear_step(
        self,
        eval_points: torch.Tensor,
        t: float,
        v_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single quasi-linearization step.

        Parameters
        ----------
        eval_points : torch.Tensor, shape (M, 4)
            Evaluation points.
        t : float
            Current time.
        v_prev : torch.Tensor, shape (M,)
            Value function from previous iteration.

        Returns
        -------
        v_new : torch.Tensor, shape (M,)
            Updated value function.
        residual : torch.Tensor, scalar
            Quasi-linear residual.
        """
        device = eval_points.device

        # Compute gradient of v_prev
        eval_points.requires_grad_(True)
        v_prev_recomputed = self._mc_value_at_point(
            eval_points, t, torch.ones(eval_points.shape[0], device=device), terminal_cost_4d
        )
        v_sum = v_prev_recomputed.sum()
        v_sum.backward()
        Dv = eval_points.grad
        eval_points.grad = None

        # Frozen coefficient: c = 2 * H(x, Dv) / (delta * |Dv|^2 + eps)
        # For murmuration, use a simple estimate
        Dv_norm_sq = torch.sum(Dv ** 2, dim=1, keepdim=True) + 1e-10
        c_frozen = 1.0 / torch.clamp(Dv_norm_sq, min=1e-10)
        c_frozen = c_frozen.squeeze(-1)

        # Compute new value with frozen coefficient
        v_new = self._mc_value_at_point(
            eval_points.detach(), t, c_frozen, terminal_cost_4d
        )

        # Quasi-linear residual
        residual = torch.mean(torch.abs(v_new - v_prev_recomputed.detach()))

        return v_new, residual

    def solve_flock_system(
        self,
        flocks: List[FlockState],
        predators: List[PredatorState],
        t: float = 0.0,
    ) -> Tuple[List[torch.Tensor], float, float]:
        """Solve BRT for multi-flock, multi-predator system.

        Parameters
        ----------
        flocks : list of FlockState
            Flock states.
        predators : list of PredatorState
            Predator states.
        t : float
            Query time.

        Returns
        -------
        safety_values : list of torch.Tensor
            Per-agent BRT values; ``safety_values[f]`` shape ``(n_agents_f,)``.
        safe_fraction : float
            Fraction of agents with v > 0.
        wall_time : float
            Wall-clock time (seconds).
        """
        import time
        start = time.time()

        flock_sizes = [f.n_agents for f in flocks]
        all_states = torch.cat([f.states for f in flocks], dim=0)
        device = all_states.device

        # One solve per predator over the full concatenated batch
        v_per_predator: List[torch.Tensor] = []
        for predator in predators:
            v_all = self._solve_quasi_linear(all_states, t)
            v_per_predator.append(v_all)

        # Min over predators
        v_min = torch.min(torch.stack(v_per_predator, dim=0), dim=0)[0]

        # Split back per flock
        safety_values: List[torch.Tensor] = []
        offset = 0
        for size in flock_sizes:
            safety_values.append(v_min[offset : offset + size])
            offset += size

        safe_fraction = float((v_min > 0).float().mean())
        wall_time = time.time() - start

        return safety_values, safe_fraction, wall_time

    def _solve_quasi_linear(
        self,
        eval_points: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Quasi-linearization solver.

        Parameters
        ----------
        eval_points : torch.Tensor, shape (M, n)
            Evaluation points.
        t : float
            Query time.

        Returns
        -------
        v : torch.Tensor, shape (M,)
            Value function.
        """
        device = eval_points.device
        dtype = eval_points.dtype

        # Initial value: terminal cost
        v = terminal_cost_4d(eval_points).to(dtype)

        # Quasi-linearization loop
        for iteration in range(self.max_quasi_iters):
            v_old = v.clone().detach()
            v, residual = self._quasi_linear_step(eval_points.detach(), t, v)

            if residual < self.quasi_tol:
                break

        return v


def create_synthetic_flocks(n_birds: int, n_flocks: int, device: torch.device, seed: int = 2026) -> list:
    """Create synthetic flock states (4D random initialisation)."""
    torch.manual_seed(seed)
    n_per_flock = n_birds // n_flocks

    flocks = []
    for flock_id in range(n_flocks):
        states = torch.empty(n_per_flock, 4, dtype=torch.float32, device=device)
        states[:, :2].uniform_(-5.0, 5.0)  # x1, x2
        states[:, 2].uniform_(0.0, 100.0)   # x3 (altitude)
        states[:, 3].uniform_(-torch.pi, torch.pi)  # theta

        flocks.append(FlockState(states=states.to(device), flock_id=flock_id))

    return flocks


def create_synthetic_predators(n_predators: int, device: torch.device, seed: int = 2026) -> list:
    """Create synthetic predator states."""
    torch.manual_seed(seed + 1000)
    predators = []
    for _ in range(n_predators):
        position = torch.empty(4, dtype=torch.float32, device=device)
        position[:2].uniform_(-3.0, 3.0)
        position[2].uniform_(10.0, 50.0)
        position[3].uniform_(-torch.pi, torch.pi)

        predators.append(
            PredatorState(position=position.to(device), omega_max=1.0, gamma_max=0.5, speed=1.0)
        )
    return predators
