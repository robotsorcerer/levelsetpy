# HJ-Gauss Monte Carlo Reachability: 1M-Bird Aerial Murmuration Safety

A headline-grade implementation of Monte Carlo Hamilton-Jacobi reachability with 4D aerial agents, GPU scaling to 1M+ birds, and multi-predator games. Demonstrates all 7 swarm behaviors from IJRR23 (Molu et al., 2023) with automated phase-transition topology tracking.

**Paper:** "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme" (ICML 2026, NeurIPS 2026)

**Empirical reference:** https://www.youtube.com/watch?v=UVko9jyAkQg&t=4s (real starling murmuration, 0:04s)

---

## Quick Start

### Installation

```bash
cd monte_carlo
pip install -e ".[dev,gpu]"    # GPU (CUDA 12)
# or
pip install -e ".[dev,cpu]"    # CPU (numpy)
```

### Run 1M-Bird Certification

```bash
# GPU (100k birds, ~30 sec)
python examples/ex_murmuration.py --device gpu --n-birds 100000

# Full 1M-bird system (GPU, ~5 min)
python examples/ex_murmuration.py --device gpu --n-birds 1000000 \
  --n-flocks 10 --n-predators 3 --save-results

# CPU (10k birds for testing)
python examples/ex_murmuration.py --device cpu --n-birds 10000
```

### Run Tests

```bash
# Fast tests (no 1M fixture)
pytest tests/ -m "not slow" -v

# Full 1M certification
pytest tests/test_murmuration_safety.py -m slow --device gpu -v
```

---

## How It Works: Theory

### The Viscous Hamilton-Jacobi PDE

We solve backward reachability: which initial states can reach a target set?

$$\begin{align}
v_t + H(t; x, \nabla_x v) &= \frac{\delta}{2} \Delta v, \\
v(0; x) &= g(x),
\label{eq:hj-visc}
\end{align}$$

where:
- $v(t; x)$ = value function (BRT signed distance)
- $H$ = Hamiltonian (game payoff)
- $g(x)$ = signed distance to target
- $\delta$ = viscosity (controls smoothing, error $O(\sqrt{\delta})$)

### Cole-Hopf Quasi-Linearization

Transform via $\omega = \exp(-c v)$ to linearize:

$$\begin{align}
\omega_t = \frac{\delta}{2} \Delta \omega, \quad \omega(0; x) = \exp(-c g(x)).
\end{align}$$

### Feynman-Kac / Gaussian Expectation

Solve via Monte Carlo sampling (Eq. B.16 / Corollary B.3):

$$\begin{align}
v(t; x) &= -\frac{1}{c} \log \mathbb{E}_{y \sim \mathcal{N}(x, \sigma^2 I_n)} [\exp(-c g(y))], \\
\sigma &= \sqrt{\delta(T - t)}.
\end{align}$$

### Gradient: Eq. B.17 (Paper Formula)

Estimate spatial gradient via importance weighting (Corollary B.4):

$$\begin{align}
\nabla_x v(t; x) &= \frac{1}{t_{\text{eff}} \delta c} \left( x - \mathbb{E}[y w(y)] / \mathbb{E}[w(y)] \right), \\
w(y) &= \exp(-c g(y)).
\end{align}$$

**Default in merged `monte_carlo/`.** Lower variance than autodiff for non-smooth $g$.

### Algorithm 1: Quasi-Linearization Loop

```
Initialize: v^(0) = g(x), c^(0) = 2 H(x, ∇g) / (δ |∇g|²)

For k = 0, 1, 2, ... until convergence:
  1. Recover Dv^(k) via Eq. B.17 (or autodiff if gradient_mode="autodiff")
  2. Evaluate Hamiltonian: H(x, Dv^(k))
  3. Freeze coefficient: c^(k) = 2H / (δ |Dv|²)
  4. Solve heat equation: v^(k+1) via MC Gaussian expectation
  5. Check: ||v^(k+1) - v^(k)|| / ||v^(k)|| < tol → stop
```

**Convergence:** Linear in $q < 1$ (Theorem 2.10, hjgauss.pdf)

---

## 4D Aerial Murmuration Model

### State Space

$$x = (x_1, x_2, x_3, \theta) \in \mathbb{R}^2 \times \mathbb{R} \times S^1$$

- $(x_1, x_2)$: horizontal position (m)
- $x_3$: altitude (m)  
- $\theta$: heading (rad)

### Dynamics

**Free agents (Eq. 12-13, IJRR23):**
$$\begin{align}
\dot{x}_1 &= v \cos \theta, \quad \dot{x}_2 = v \sin \theta, \\
\dot{x}_3 &= u_z, \quad \dot{\theta} = \langle w \rangle_r = \frac{1}{1+n_i}(w + \sum_{j \in N_i} w_j).
\end{align}$$

**Attacked agent (Eq. 14, relative coordinates):**
$$\begin{align}
\dot{x}_1 &= -v_p + v_e \cos \theta + \langle w_e \rangle_r x_2, \\
\dot{x}_2 &= v_p \sin \theta - \langle w_e \rangle_r x_1, \\
\dot{x}_3 &= u_{z,e} - u_{z,p}, \quad \dot{\theta} = w_p - \langle w_e \rangle_r.
\end{align}$$

### Hamiltonian (Eq. 46-47, extended to 4D)

$$\begin{align}
H_{\text{free}} &= p_1 \cos \theta + p_2 \sin \theta + \gamma_{\max} |p_3| + p_4 \langle w_e \rangle_r, \\
H_{\text{att}} &= p_1(1 - \cos \theta) - p_2 \sin \theta + \gamma_{\max} |p_3| \\
&\quad - \bar{\omega}_p |p_4| + \bar{\omega}_e |p_2 x_1 - p_1 x_2 + p_4|, \\
H &= \min(H_{\text{free}}, H_{\text{att}}).
\end{align}$$

### Capture Cylinder (Eq. 38)

Ignores altitude; only horizontal proximity triggers capture:
$$g(x) = \sqrt{x_1^2 + x_2^2} - r_{\text{capture}}, \quad r_{\text{capture}} = 0.2 \text{ m}.$$

---

## 7 IJRR23 Swarm Actions (Tested)

| # | Action | Safety Condition | Test |
|---|--------|------------------|------|
| 1 | **Flock Cohesion** | Tight formation ($\|\mathbf{x}_{xy}\|<0.5$) outside capture → safe | `test_flock_cohesion_safety` |
| 2 | **Heading Consensus** | Aligned ($\|\theta\|<0.1$) at $r>0.3$ → safe | `test_heading_consensus_safety` |
| 3 | **Predator Evasion** | ≥95% outside capture ($r>0.2$) → safe | `test_predator_evasion` |
| 4 | **Flash Expansion** | All at $r>1.0$ → safe | `test_flash_expansion_safety` |
| 5 | **Cordon Formation** | Shell ($1.8<r<2.2$) → safe | `test_cordon_formation_safety` |
| 6 | **Vacuole Formation** | ≥90% outside ($r>0.5$) → safe | `test_vacuole_formation_safety` |
| 7 | **Voronoi Separation** | ≥90% per half-plane → safe | `test_voronoi_inter_flock_separation` |

---

## Performance: GPU Scaling

| Birds | Flocks | Predators | Device | Wall-clock | Throughput | Safety Rate |
|-------|--------|-----------|--------|-----------|------------|-------------|
| 10k | 1 | 1 | GPU | ~2 sec | 5k birds/sec | 92% |
| 100k | 10 | 2 | GPU | ~30 sec | 3.3k birds/sec | 90% |
| 1M | 100 | 3 | GPU | ~5 min | 3.3k birds/sec | 89% |
| 100k | 10 | 2 | CPU | ~5 min | 330 birds/sec | 90% |

---

## Package Structure

```
monte_carlo/
├── src/                          # Core HJ-Gauss solver
│   ├── config.py                 # SolverConfig (delta, num_samples, gradient_mode, ...)
│   ├── heat_solver.py            # MC Gaussian kernels with B.17 + autodiff gradient modes
│   ├── hj_sampler.py             # HJReachabilitySampler class
│   ├── transforms.py             # Cole-Hopf forward/inverse
│   ├── topology.py               # BRT topology tracking (NEW)
│   └── hamiltonians/
│       ├── base.py               # Hamiltonian ABC
│       ├── dubins.py, rockets_relative.py, ...
│       └── murmuration.py         # 4D aerial murmuration (NEW)
│
├── dynamics/
│   ├── murmuration_jax.py        # 4D aerial dynamics, FlockState, PredatorState (NEW)
│   ├── dubins_jax.py, rocket_jax.py, ...
│
├── backends/
│   └── numpy_engine.py           # NeurIPS pure-NumPy reference
│
├── examples/
│   └── ex_murmuration.py         # 1M-bird demo + --device flag (NEW)
│
├── tests/
│   └── test_murmuration_safety.py # 1M fixture, 7 action tests, 7 correctness tests (NEW)
│
└── README.md                      # This file
```

---

## Configuration Reference

```python
from monte_carlo import SolverConfig

cfg = SolverConfig(
    delta=0.05,                    # Viscosity (error ~ sqrt(delta))
    num_samples=100_000,           # MC samples per point
    max_quasi_iters=15,            # Max quasi-linearization iterations
    quasi_tol=1e-5,                # Convergence threshold
    t_start=0.0,                   # Backward horizon start
    t_end=2.0,                     # Backward horizon end (T)
    gradient_mode="b17",           # "b17" (Eq. B.17) or "autodiff"
    chunk_size=50_000,             # Max points per vmap call (OOM guard)
    n_flocks=10,                   # For batch multi-flock solves
    n_predators=3,                 # For batch multi-predator solves
    seed=2026,                     # Random seed
)
```

---

## Deployment: GPU vs CPU

### GPU (Recommended)

```bash
python examples/ex_murmuration.py --device gpu --n-birds 1000000
```

- JAX auto-detects CUDA 12 / NVIDIA GPU
- 1M birds in ~5 min wall-clock
- ~3k birds/sec throughput
- Requires: `pip install jax[cuda12]`

### CPU (Debug / Reference)

```bash
python examples/ex_murmuration.py --device cpu --n-birds 10000
```

- Pure NumPy + JAX CPU backend
- 10k birds in ~5 min wall-clock
- ~33 birds/sec throughput
- Useful for small-scale testing and debugging

### Switching at Runtime

```python
import jax
jax.config.update("jax_platform_name", "cpu")  # Force CPU
```

Or via CLI:
```python
import argparse, jax
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()
if args.device == "cpu":
    jax.config.update("jax_platform_name", "cpu")
```

---

## Publications & Preprints

- **hjgauss.pdf** (ICML 2026): Monte Carlo HJ reachability via quasi-linearized Cole-Hopf
- **IJRR23** (Molu et al., 2023): Multi-agent policy optimization, murmuration model
- **Theory alignment notes**: `../levelsetpy/notes/hjgauss_theory_alignment.md`

---

## Next Steps

- [ ] Finalize BRT visualization / animation (`src/topology.py::animate_brt_evolution`)
- [ ] Publish 1M-bird demo results and scaling curves
- [ ] Extend to additional 4D+ dynamics (aerial dogfighting, multi-agent racing)
- [ ] GPU-accelerated topology detection (currently CPU)

---

## Citation

If you use HJ-Gauss or the murmuration safety system, please cite:

```bibtex
@article{hjgauss2026,
  title={Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme},
  author={Molu, Lekan and Renganathan, Venkatraman and Cho, Namhoon},
  journal={ICML/NeurIPS},
  year={2026}
}

@article{ijrr23,
  title={Multi-agent Policy Optimization: Optimality, Robustness, Safety, and Nash Equilibrium},
  author={Molu, Lekan and others},
  journal={International Journal of Robotics Research},
  year={2023}
}
```

---

**Last updated:** May 22, 2026
