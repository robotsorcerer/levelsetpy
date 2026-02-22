# Sampling-Based HJ Reachability - Implementation

This directory contains implementations of the **quasi-linearization algorithm with Cole-Hopf transformation** for computing Hamilton-Jacobi reachability as described in the  paper: [Scalable Approximately Correct HJ Reachability via Importance Sampling](#).


<div align="center">
 <img src="results/dubins_simple3d.png" height="250px" width="250px">
 <img src="results/sphere2_2d.jpeg" height="250px" width="250px">
 <img src="results/sphere_union_2d.jpeg" height="250px" width="250px">
 <img src="results/sph_rect_diff_2d.jpeg" height="250px" width="250px">
</div>

## Algorithm Overview

**Algorithm 1: Quasi-Linearization with Cole-Hopf Transformation**

The algorithm solves the viscous Hamilton-Jacobi PDE:

$$
  		\bm{v}_t + \bm{H}(t; \bm{x}, D \bm{v}) = 0 \text{   in   } \Omega \times (0, T], \ \\ 
		\bm{v}(0; \bm{x}) = \bm{g}(0; \bm{x}) \text{   on   } \partial \Omega \times \{t=0\}
$$


where $\bm{v}$ is the value, $\bm{x}$ is the state, $\bm{H}$ is the system's Hamiltonian function, and $\Omega$ is an open set in $\mathbb{R}^n$. For a $\delta > 0$, the _vanishing viscosity solution_ to the above equation is 

$$
		\bm{v}_t^\delta + \bm{H}(t; \bm{x}, D \bm{v}^\delta) = \dfrac{\delta}{2} \Delta \bm{v}^\delta \text{ in } \mathbb{R}^n \times (0, T];  \\
		\bm{v}^\delta(0; \bm{x}) = \bm{g}(0; \bm{x}) \text{ on } \mathbb{R}^n \times \{t=0\}.
$$

Let $\bm{H}^\delta:=\bm{H}(t; \bm{x}, D \bm{v}^\delta)$ and define 

$$
	\bm{c}(t; \bm{x}) = \frac{2}{\delta}\cdot\bm{H}^\delta/|D\bm{v}^\delta|^2.
$$

Using the Cole-Hopf transformation $\bm{\omega}^\delta:= \exp(-\bm{c} \bm{\omega}^\delta)$, we linearize to the HJ equation (please see the paper) and iterate:

1. **Initialize**: $\bm{v}^{(0)} = \bm{g}(\bm{x})$, $\bm{c}^{(0)} = \frac{2}{\delta}\cdot\bm{H}^\delta/|D\bm{g}|^2.$  

2. **Fix**: Convergence parameter $\epsilon > 0$;

3. **For k = 0, 1, 2, ...**:
   - Freeze coefficient $\bm{c}^(k)$ at current iterate;

   - Solve Linear-HJ  equation via the Gaussian expectation $\bm{\omega}^{(k)}(0; \bm{x})
 = \text{exp}(-\bm{c}^{(k)} \bm{g}(\bm{x}))$;

    - Recover $\bm{v}^{(k+1)} = (-1/\bm{c}) \, \text{log} \, \bm \omega^{(k+1)}$;

   - Update gradient $D\bm{v}^{(k+1)}$ and coefficient $\bm{c}^{(k+1)} = \frac{2 \bm{H}(t; \bm{x}, D\bm{v}^{(k+1)})}{\delta |D\bm{v}^{(k+1)|^2}}$;

   - Check convergence $\|\bm{v}^{(k+1)} - \bm{v}^{(k)} \|/\| \bm{v}^{(k)}\| < \epsilon$.


## Dependencies

```bash
pip install numpy scipy matplotlib pytest
pip install jax jaxlib  # Optional, for JAX implementations
```

## File Structure

```
reach_sample/
├── utils.py                     # Common utilities (Cole-Hopf, MC sampling)
├── rocket_python.py             # Rocket implementation (NumPy)
├── rocket_jax.py                # Rocket implementation (JAX)
├── rocket_test.py               # Comprehensive tests for rocket
├── dubins_python.py             # Dubins car implementation (NumPy)
├── dubins_jax.py                # Dubins car implementation (JAX)
├── dubins_test.py               # Comprehensive tests for Dubins
├── demo_rocket.py               # Rocket visualization demo
├── demo_dubins.py               # Dubins visualization demo
├── compare_rocket_levelsetpy.py # Comparison with levelsetpy
├── compare_dubins_levelsetpy.py # Comparison with levelsetpy
└── README.md                    # This file
```

## Examples

### 1. Rocket Vehicle (Relative 3D Coordinates)

**System**: Two rockets in relative coordinates

$$
dx/dt = a·cos(\theta) \\
dy/dt = (g - a)·sin(\theta) + u_y \\
d\theta/dt = 0
$$

**Hamiltonian**:
$$
\bm{H} = -a·cos(\theta)·p_1 - sin(\theta)(g - a - u_y)·p_2 + |u|(|p_1 + p_2| - |p_1 - p_2|)
$$

**Terminal cost**: Signed distance to cylinder of radius 1.5


## Configuration

**SolverConfig** parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `delta` | Viscosity parameter | 0.1 |
| `num_samples` | MC samples per iteration | 10,000 |
| `max_iters` | Maximum quasi-linearization iterations | 20 |
| `tol` | Relative residual tolerance | 1e-3 |
| `t_start` | Start time | 0.0 |
| `t_end` | End time | 1.0 |
| `seed` | Random seed | 42 |

**Run**:
```bash
# Tests
python -m pytest rocket_test.py -v

# Demo visualization
python demo_rocket.py

# Compare with levelsetpy
python compare_rocket_levelsetpy.py
```

### 2. Dubins Car (3D)

**System**: Dubins car with heading

$$
dx/dt = v·cos(\theta)
dy/dt = v·sin(\theta)
d\theta/dt = \omega
$$

**Two modes**:

- **Simple**: Single Dubins car reaching target
  - $\bm{H} = v·cos(\theta)·p_1 + v·sin(\theta)·p_2 - \omega_{max}|p_3|$

- **Pursuit-Evasion**: Relative Dubins (Merz 1972)
  - Pursuer vs Evader differential game
  - $\bm{H} = -v_p·cos(\theta)·p_1 - v_p·sin(\theta)·p_2 - v_e + \omega(|p_1·sin(\theta) - p_2·cos(\theta)| - |p_1·sin(\theta) + p_2·cos(\theta)|)$

**Terminal cost**: Signed distance to cylinder of radius 0.5

**Run**:
```bash
# Tests
python -m pytest dubins_test.py -v

# Demo visualization
python demo_dubins.py

# Compare with levelsetpy
python compare_dubins_levelsetpy.py
```

## Output

Visualizations are saved to `results/`:
- `rocket_brt_slices.png` - Value functions at different \theta
- `rocket_convergence.png` - Convergence history
- `rocket_3d_value.png` - 3D value function
- `dubins_simple_slices.png` - Simple mode BRT
- `dubins_pursuit_evasion_slices.png` - PE mode BRT
- `dubins_mode_comparison.png` - Side-by-side comparison

## Algorithm Properties

- Value function is **negative inside target**, **positive outside**
- Zero-level set defines the **Backward Reachable Tube (BRT)**
- Residuals **decrease monotonically** with iterations
- Works for **arbitrary Hamiltonians** (not just convex)
- **Scalable** to high dimensions (tested up to 3D, extensible)

## Comparison with [levelsetpy](https://github.com/robotsorcerer/levelsetpy)

The comparison scripts validate that:
1. Zero-level sets match levelsetpy results
2. Value function magnitudes are comparable
3. BRT structure is preserved
4. Computational times are competitive

See `compare_rocket_levelsetpy.py` and `compare_dubins_levelsetpy.py` for details.

## References

- Paper: "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme" (ICML 2026)
- Algorithm 1: Section 3.3 (Quasi-Linearization Algorithm Cole-Hopf)
- levelsetpy: [levelsetpy](https://github.com/robotsorcerer/levelsetpy)


## Notes

- JAX implementations provide automatic differentiation and GPU acceleration
- NumPy implementations are easier to debug and understand
- All tests pass with NumPy; JAX tests skipped if JAX unavailable
- Grid resolution affects accuracy vs. computation time trade-off
- Number of MC samples affects variance vs. computation time trade-off

---

**Author**: Lekan Molu's Implementation based on ICML 2026 paper
**Date**: January/February 2026
**Status**: ✅ Verified and tested

## ToDos


### Future Enhancements
- [ ] Add GPU acceleration (JAX on GPU)
- [ ] Implement adaptive MC sampling
- [ ] Extend to higher dimensions (4D, 5D)
- [ ] Add reach-avoid problems
- [ ] Implement time-varying Hamiltonians
- [ ] Compare with DeepReach
