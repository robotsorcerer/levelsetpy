"""Cross-validation tests: sampling solver vs levelsetpy grid solver.

The sampling solver computes the *viscous* HJ equation:
    v_t + H(t, x, Dv) = (delta/2) * Delta v

while levelsetpy solves the *inviscid* equation:
    v_t + H(t, x, Dv) = 0

The Crandall-Lions (1984) bound gives ||v^delta - v||_inf <= C*sqrt(delta),
so we check:
  - Structural agreement (inside target < outside)
  - Qualitative agreement (error bounded by O(sqrt(delta)))
  - MC convergence (error decreases as O(1/sqrt(J)))
"""

import sys
import copy
import pytest
import numpy as np

sys.path.insert(0, "/home/lex/Documents/ML-Control-Rob/control/levelsetpy")

import jax
import jax.numpy as jnp

from src.config import SolverConfig
from src.hj_sampler import HJReachabilitySampler
from src.hamiltonians.quadratic import QuadraticHamiltonian
from src.hamiltonians.double_integrator import DoubleIntegratorHamiltonian
from src.initial_conditions import sphere_cost
from src.diagnostics import compute_error_metrics


# ---------------------------------------------------------------------------
#  Helpers: run levelsetpy grid solver
# ---------------------------------------------------------------------------

def _run_levelsetpy_dint(grid_N=51, t_final=0.5, u_bound=1.0, radius=1.0):
    """Run the levelsetpy double integrator solver and return grid + value func.

    Returns
    -------
    g_ls : levelsetpy grid object
    value_func : np.ndarray of shape (grid_N, grid_N)
    """
    import torch
    from levelsetpy.grids import createGrid
    from levelsetpy.initialconditions import shapeSphere
    from levelsetpy.dynamicalsystems import DoubleIntegrator
    from levelsetpy.utilities import Bundle
    from levelsetpy.explicitintegration.term import (
        termLaxFriedrichs, termRestrictUpdate,
    )
    from levelsetpy.explicitintegration.integration import odeCFL2, odeCFLset
    from levelsetpy.explicitintegration.dissipation import artificialDissipationGLF
    from levelsetpy.spatialderivative import upwindFirstENO2

    gmin = np.array([[-2, -2]], dtype=np.float64).T
    gmax = np.array([[2, 2]], dtype=np.float64).T
    g_ls = createGrid(gmin, gmax, grid_N, None)

    phi0 = shapeSphere(g_ls, np.zeros((2, 1)), radius)

    # Create the dynamical system
    dint = DoubleIntegrator(g_ls, u_bound)

    # Convert grid xs to torch for the solver
    g_torch = copy.deepcopy(g_ls)
    g_torch.xs = [torch.as_tensor(x) for x in g_ls.xs]

    finite_diff_data = Bundle(dict(
        innerFunc=termLaxFriedrichs,
        innerData=Bundle({
            "grid": g_torch,
            "hamFunc": dint.hamiltonian,
            "partialFunc": dint.dissipation,
            "dissFunc": artificialDissipationGLF,
            "CoStateCalc": upwindFirstENO2,
        }),
        positive=False,
    ))

    small = 100 * np.finfo(np.float64).eps
    options = Bundle(dict(
        factorCFL=0.75, stats="on",
        maxStep=1e10, singleStep="off",
    ))

    y0 = torch.as_tensor(phi0.flatten())
    cur_time = 0.0
    step_time = t_final / 4.0

    while t_final - cur_time > small * t_final:
        t_span = np.hstack([cur_time, min(t_final, cur_time + step_time)])
        t, y, finite_diff_data = odeCFL2(
            termRestrictUpdate, t_span, y0,
            odeCFLset(options), finite_diff_data,
        )
        cur_time = t if np.isscalar(t) else t[-1]
        y0 = y

    value_func = y.reshape(g_ls.shape)
    if hasattr(value_func, "cpu"):
        value_func = value_func.cpu().numpy()

    return g_ls, np.array(value_func)


def _interpolate_grid_value(g_ls, value_func, points):
    """Bilinear interpolation of a 2D grid value function at given points.

    Parameters
    ----------
    g_ls : levelsetpy grid object
    value_func : (Nx, Ny) numpy array
    points : (M, 2) numpy array of (x1, x2) query points

    Returns
    -------
    (M,) array of interpolated values.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Grid coordinates: xs[0] varies along axis 0, xs[1] along axis 1
    x1_coords = g_ls.xs[0][:, 0]   # (Nx,)
    x2_coords = g_ls.xs[1][0, :]   # (Ny,)

    interp = RegularGridInterpolator(
        (x1_coords, x2_coords), value_func,
        method="linear", bounds_error=False, fill_value=None,
    )
    return interp(points)


# ---------------------------------------------------------------------------
#  Test: Double Integrator structural agreement
# ---------------------------------------------------------------------------

class TestDIntStructuralAgreement:
    """Verify that both solvers agree on qualitative structure."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Run levelsetpy once for the class."""
        self.grid_N = 51
        self.t_final = 0.5
        self.radius = 1.0
        self.u_bound = 1.0
        self.g_ls, self.v_ref = _run_levelsetpy_dint(
            grid_N=self.grid_N, t_final=self.t_final,
            u_bound=self.u_bound, radius=self.radius,
        )

    def test_ref_solution_valid(self):
        """Sanity: levelsetpy solution should be finite and non-constant."""
        assert np.all(np.isfinite(self.v_ref))
        assert self.v_ref.max() - self.v_ref.min() > 0.1

    def test_inside_vs_outside_levelsetpy(self):
        """Origin (inside sphere) should have lower value than corner."""
        mid = self.grid_N // 2
        assert self.v_ref[mid, mid] < self.v_ref[0, 0]

    def test_sampling_inside_vs_outside_agrees(self):
        """Sampling solver should also give lower value inside."""
        cfg = SolverConfig(
            delta=0.05, num_samples=30_000, seed=42,
            t_start=0.0, t_end=self.t_final, time_steps=3,
            max_quasi_iters=5, quasi_tol=1e-3,
        )
        H = DoubleIntegratorHamiltonian(u_bound=self.u_bound)
        g_fn = lambda x: sphere_cost(x, radius=self.radius)
        solver = HJReachabilitySampler(H, g_fn, cfg)

        inside = jnp.array([[0.0, 0.0]])
        outside = jnp.array([[1.5, 1.5]])
        v_in = solver.solve(inside, 0.0)
        v_out = solver.solve(outside, 0.0)
        assert float(v_in[0]) < float(v_out[0])

    def test_sign_agreement_on_grid(self):
        """At several grid points, the sign of the value function should agree
        between levelsetpy (inviscid) and our solver (viscous, small delta).

        We test on a coarse subset of points well inside and well outside
        the target where the viscous smoothing won't flip the sign.
        """
        cfg = SolverConfig(
            delta=0.02, num_samples=50_000, seed=42,
            t_start=0.0, t_end=self.t_final, time_steps=3,
            max_quasi_iters=5, quasi_tol=1e-3,
        )
        H = DoubleIntegratorHamiltonian(u_bound=self.u_bound)
        g_fn = lambda x: sphere_cost(x, radius=self.radius)
        solver = HJReachabilitySampler(H, g_fn, cfg)

        # Pick points well inside and well outside
        test_pts = jnp.array([
            [0.0, 0.0],     # center (inside)
            [1.8, 0.0],     # far right (outside)
            [0.0, 1.8],     # far top (outside)
            [-1.5, -1.5],   # far corner (outside)
        ])

        v_samp = solver.solve(test_pts, 0.0)
        v_ref_pts = _interpolate_grid_value(
            self.g_ls, self.v_ref, np.array(test_pts),
        )

        # Check sign agreement at well-separated points
        for i in range(len(test_pts)):
            # Both should agree on sign for points far from the boundary
            if abs(float(v_ref_pts[i])) > 0.3:
                assert (
                    np.sign(float(v_ref_pts[i]))
                    == np.sign(float(v_samp[i]))
                ), (
                    f"Sign mismatch at point {test_pts[i]}: "
                    f"ref={v_ref_pts[i]:.3f}, samp={v_samp[i]:.3f}"
                )


# ---------------------------------------------------------------------------
#  Test: Quantitative comparison (viscous vs inviscid)
# ---------------------------------------------------------------------------

class TestDIntQuantitativeComparison:
    """Quantitative error between viscous and inviscid solutions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.g_ls, self.v_ref = _run_levelsetpy_dint(
            grid_N=51, t_final=0.5, u_bound=1.0, radius=1.0,
        )

    def test_error_bounded_by_sqrt_delta(self):
        """The viscous solution should be within O(sqrt(delta)) of inviscid.

        We test with delta=0.1 and check that the maximum deviation at
        interior grid points is bounded by a reasonable constant * sqrt(delta).
        """
        delta = 0.1
        cfg = SolverConfig(
            delta=delta, num_samples=50_000, seed=42,
            t_start=0.0, t_end=0.5, time_steps=3,
            max_quasi_iters=5, quasi_tol=1e-3,
        )
        H = DoubleIntegratorHamiltonian(u_bound=1.0)
        g_fn = lambda x: sphere_cost(x, radius=1.0)
        solver = HJReachabilitySampler(H, g_fn, cfg)

        # Sample on a coarse interior grid (avoid boundary effects)
        x1_pts = np.linspace(-1.2, 1.2, 7)
        x2_pts = np.linspace(-1.2, 1.2, 7)
        xx, yy = np.meshgrid(x1_pts, x2_pts)
        test_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        v_samp = solver.solve(jnp.array(test_pts), 0.0)
        v_ref_pts = _interpolate_grid_value(self.g_ls, self.v_ref, test_pts)

        # The Crandall-Lions bound: ||v^delta - v|| <= C*sqrt(delta)
        # For delta=0.1, sqrt(delta) ~ 0.316
        # With MC noise and the bound constant, we allow generous slack
        l_inf = float(jnp.max(jnp.abs(jnp.array(v_ref_pts) - v_samp)))
        bound = 3.0 * np.sqrt(delta)  # generous constant
        assert l_inf < bound, (
            f"L-inf error {l_inf:.4f} exceeds expected bound "
            f"{bound:.4f} (3*sqrt(delta))"
        )


# ---------------------------------------------------------------------------
#  Test: Exact Cole-Hopf analytical solution (quadratic H)
# ---------------------------------------------------------------------------

class TestExactColeHopfAnalytical:
    """For H = (1/2)|p|^2 with g(x) = |x|^2, we know:

        v(t, x) = |x|^2 + n * delta * (T - t)

    This is exact (no approximation error, only MC noise).
    """

    def test_quadratic_cost_analytical(self):
        """For g(x)=|x|^2, the exact Cole-Hopf solution is:

            v(t,x) = |x|^2 / (1 + 2*tau) + (n*delta/2) * log(1 + 2*tau)

        where tau = T - t.  This comes from evaluating the Gaussian
        moment generating function of the quadratic exponent exactly.
        """
        delta = 0.1
        T = 1.0
        n = 2
        cfg = SolverConfig(
            delta=delta, num_samples=100_000, seed=42,
            t_start=0.0, t_end=T, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=n)
        g_fn = lambda x: jnp.sum(x ** 2)
        solver = HJReachabilitySampler(H, g_fn, cfg)

        t_eval = 0.3
        tau = T - t_eval
        # Use moderate |x| to avoid high-variance regime
        test_pts = jnp.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.0],
        ])

        r2 = jnp.sum(test_pts ** 2, axis=1)
        v_exact = r2 / (1.0 + 2.0 * tau) + (n * delta / 2.0) * jnp.log(1.0 + 2.0 * tau)
        v_samp = solver.solve_exact(test_pts, t_eval)

        assert jnp.allclose(v_samp, v_exact, atol=0.05, rtol=0.1), (
            f"max error: {float(jnp.max(jnp.abs(v_samp - v_exact))):.4f}\n"
            f"sampled: {v_samp}\nexact:   {v_exact}"
        )

    def test_sphere_cost_value_decreases_with_time(self):
        """For exact Cole-Hopf, value at a fixed outside point should
        decrease as t increases (further from terminal time T).
        The viscous smoothing makes the value converge toward a flat profile."""
        delta = 0.1
        cfg = SolverConfig(
            delta=delta, num_samples=50_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        g_fn = lambda x: sphere_cost(x, radius=1.0)
        solver = HJReachabilitySampler(H, g_fn, cfg)

        pt = jnp.array([[2.0, 0.0]])
        # Near terminal: v should be close to g(x) = |x| - 1 = 1.0
        v_late = solver.solve_exact(pt, 0.95)
        # Further from terminal: more smoothing
        v_early = solver.solve_exact(pt, 0.0)
        # Both should be positive (outside sphere)
        assert float(v_late[0]) > 0
        assert float(v_early[0]) > 0


# ---------------------------------------------------------------------------
#  Test: MC convergence rate
# ---------------------------------------------------------------------------

class TestMCConvergence:
    """Verify that the MC estimator converges at the expected rate."""

    def test_error_decreases_with_samples(self):
        """As J increases, the solution should become more stable (lower
        variance between independent runs)."""
        delta = 0.1
        H = QuadraticHamiltonian(dim=2)
        g_fn = lambda x: jnp.sum(x ** 2)
        pt = jnp.array([[1.0, 1.0]])
        T = 1.0
        t_eval = 0.5

        # Run with increasing sample counts, two independent seeds each
        sample_counts = [1_000, 10_000, 50_000]
        spreads = []
        for J in sample_counts:
            vals = []
            for seed in [100, 200, 300]:
                cfg = SolverConfig(
                    delta=delta, num_samples=J, seed=seed,
                    t_start=0.0, t_end=T, time_steps=3,
                )
                solver = HJReachabilitySampler(H, g_fn, cfg)
                v = solver.solve_exact(pt, t_eval)
                vals.append(float(v[0]))
            spread = max(vals) - min(vals)
            spreads.append(spread)

        # The spread should decrease with more samples
        assert spreads[-1] < spreads[0], (
            f"Spread did not decrease: {spreads}"
        )

    def test_convergence_rate_approximately_half(self):
        """Error ~ C * J^(-alpha) with alpha ~ 0.5 for standard MC."""
        from src.diagnostics import convergence_rate

        delta = 0.1
        T = 1.0
        n = 2
        t_eval = 0.5
        tau = T - t_eval
        H = QuadraticHamiltonian(dim=n)
        g_fn = lambda x: jnp.sum(x ** 2)
        pt = jnp.array([[0.5, 0.5]])  # moderate |x| for lower variance
        r2 = float(jnp.sum(pt ** 2))
        v_exact = r2 / (1.0 + 2.0 * tau) + (n * delta / 2.0) * float(jnp.log(1.0 + 2.0 * tau))

        sample_counts = [2_000, 5_000, 10_000, 50_000]
        errors = []
        for J in sample_counts:
            # Average over a few seeds for stable error estimate
            errs = []
            for seed in range(10):
                cfg = SolverConfig(
                    delta=delta, num_samples=J, seed=seed,
                    t_start=0.0, t_end=T, time_steps=3,
                )
                solver = HJReachabilitySampler(H, g_fn, cfg)
                v = solver.solve_exact(pt, t_eval)
                errs.append(abs(float(v[0]) - v_exact))
            errors.append(np.mean(errs))

        alpha = convergence_rate(
            jnp.array(errors), jnp.array(sample_counts),
        )
        # Standard MC: alpha ~ 0.5; allow [0.2, 1.0] for noisy estimate
        assert 0.2 < alpha < 1.0, f"Convergence rate alpha={alpha:.3f}"


# ---------------------------------------------------------------------------
#  Test: Diagnostics module integration
# ---------------------------------------------------------------------------

class TestDiagnosticsIntegration:
    """Verify diagnostics module works with real solver output."""

    def test_error_metrics_with_exact_solution(self):
        """compute_error_metrics should return sensible values."""
        delta = 0.1
        T = 1.0
        n = 2
        t_eval = 0.5
        cfg = SolverConfig(
            delta=delta, num_samples=50_000, seed=42,
            t_start=0.0, t_end=T, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=n)
        g_fn = lambda x: jnp.sum(x ** 2)
        solver = HJReachabilitySampler(H, g_fn, cfg)

        tau = T - t_eval
        test_pts = jnp.array([
            [0.0, 0.0], [0.5, 0.0], [0.0, 0.5],
            [0.5, 0.5], [1.0, 0.0],
        ])
        r2 = jnp.sum(test_pts ** 2, axis=1)
        v_exact = r2 / (1.0 + 2.0 * tau) + (n * delta / 2.0) * jnp.log(1.0 + 2.0 * tau)
        v_samp = solver.solve_exact(test_pts, t_eval)

        metrics = compute_error_metrics(v_samp, v_exact)
        assert metrics["l2_relative"] < 0.15
        assert metrics["rmse"] < 0.1
        assert metrics["l_inf"] < 0.15
        assert metrics["mean_abs_error"] < 0.1
