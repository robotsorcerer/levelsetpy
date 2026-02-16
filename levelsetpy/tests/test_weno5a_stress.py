"""
Rigorous stress tests for WENO5a spatial derivative scheme.

Tests convergence order, polynomial exactness, behavior near discontinuities,
grid configurations, edge cases, and mathematical symmetry properties.
"""

import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.boundarycondition import addGhostPeriodic, addGhostExtrapolate
from levelsetpy.spatialderivative import (
    upwindFirstFirst,
    upwindFirstENO2,
    upwindFirstENO3a,
    upwindFirstWENO5a,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_periodic_grid(N, dim):
    """Create a periodic grid on [-1, 1]^dim with N nodes per dimension."""
    gmin = -1.0 * np.ones((dim, 1), dtype=np.float64)
    gmax = +1.0 * np.ones((dim, 1), dtype=np.float64)
    Narr = N * np.ones((dim, 1), dtype=np.int64)
    pdDims = list(range(dim))
    g = createGrid(gmin, gmax, Narr, pdDims, process=True)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


def _interior_slice(g, margin=6):
    """Return a tuple of slices that skip `margin` points from each edge."""
    return tuple(slice(margin, s - margin) for s in g.N.flatten())


# ---------------------------------------------------------------------------
# 1. Convergence Tests
# ---------------------------------------------------------------------------

class TestWENO5aConvergence:
    """Verify order of accuracy on smooth functions."""

    def test_convergence_rate_smooth_2d(self):
        """Successive grid refinement should show >= 4th order convergence."""
        resolutions = [21, 41, 81, 161]
        errors = []
        for N in resolutions:
            g = _make_periodic_grid(N, 2)
            data = (torch.sin(pi * g.xs[0]) * torch.cos(pi * g.xs[1])).to(torch.float64)
            exact = (pi * torch.cos(pi * g.xs[0]) * torch.cos(pi * g.xs[1])).to(torch.float64)
            derivL, derivR = upwindFirstWENO5a(g, data, 0)
            centered = 0.5 * (derivL + derivR)
            sl = _interior_slice(g, margin=6)
            err = torch.abs(centered[sl] - exact[sl]).max().item()
            errors.append(err)

        # Check convergence ratios between successive refinements
        for i in range(len(errors) - 1):
            if errors[i + 1] < 1e-14:
                continue  # already at machine precision
            ratio = errors[i] / errors[i + 1]
            # 5th order ideal ratio ~32 (2^5), require >= 16 (4th order)
            assert ratio > 16, (
                f"Convergence ratio {ratio:.2f} at N={resolutions[i]}->{resolutions[i+1]}, "
                f"errors={errors[i]:.2e}->{errors[i+1]:.2e}"
            )

    def test_polynomial_exactness_degree_0_to_4(self):
        """5th-order scheme is exact on polynomials up to degree 4.

        On periodic grids, higher-degree polynomials have jumps at the boundary
        that affect nearby stencils, so we use generous interior margins and
        tolerances that tighten with grid refinement for degree 4.
        """
        g = _make_periodic_grid(81, 2)
        sl = _interior_slice(g, margin=10)
        x = g.xs[0].to(torch.float64)

        polys = [
            (torch.ones_like(x), torch.zeros_like(x), "degree 0", 1e-10),
            (x, torch.ones_like(x), "degree 1", 1e-8),
            (x ** 2, 2 * x, "degree 2", 1e-6),
            (x ** 3, 3 * x ** 2, "degree 3", 1e-4),
            (x ** 4, 4 * x ** 3, "degree 4", 1e-2),
        ]
        for data, exact_deriv, label, tol in polys:
            derivL, derivR = upwindFirstWENO5a(g, data, 0)
            centered = 0.5 * (derivL + derivR)
            err = torch.abs(centered[sl] - exact_deriv[sl]).max().item()
            assert err < tol, f"Polynomial {label}: max error {err:.2e} > tol {tol:.0e}"

    def test_convergence_rate_1d(self):
        """Convergence study on a 1D grid."""
        resolutions = [41, 81, 161]
        errors = []
        for N in resolutions:
            g = _make_periodic_grid(N, 1)
            data = torch.sin(2 * pi * g.xs[0]).to(torch.float64)
            exact = (2 * pi * torch.cos(2 * pi * g.xs[0])).to(torch.float64)
            derivL, derivR = upwindFirstWENO5a(g, data, 0)
            centered = 0.5 * (derivL + derivR)
            sl = _interior_slice(g, margin=6)
            err = torch.abs(centered[sl] - exact[sl]).max().item()
            errors.append(err)

        for i in range(len(errors) - 1):
            if errors[i + 1] < 1e-14:
                continue
            ratio = errors[i] / errors[i + 1]
            assert ratio > 16, (
                f"1D convergence ratio {ratio:.2f}, "
                f"errors={errors[i]:.2e}->{errors[i+1]:.2e}"
            )

    def test_higher_order_than_eno3a(self):
        """WENO5a should have smaller error than ENO3a on smooth data."""
        for N in [41, 81]:
            g = _make_periodic_grid(N, 2)
            data = (torch.sin(pi * g.xs[0]) * torch.cos(pi * g.xs[1])).to(torch.float64)
            exact = (pi * torch.cos(pi * g.xs[0]) * torch.cos(pi * g.xs[1])).to(torch.float64)
            sl = _interior_slice(g, margin=6)

            dL_w, dR_w = upwindFirstWENO5a(g, data, 0)
            err_weno = torch.abs(0.5 * (dL_w + dR_w)[sl] - exact[sl]).max().item()

            dL_e, dR_e = upwindFirstENO3a(g, data, 0)
            err_eno = torch.abs(0.5 * (dL_e + dR_e)[sl] - exact[sl]).max().item()

            assert err_weno < err_eno, (
                f"WENO5a error {err_weno:.2e} >= ENO3a error {err_eno:.2e} at N={N}"
            )


# ---------------------------------------------------------------------------
# 2. Discontinuity Tests
# ---------------------------------------------------------------------------

class TestWENO5aDiscontinuities:
    """Verify non-oscillatory behavior near discontinuities."""

    def test_step_function_no_gibbs(self):
        """Step function: no oscillation away from the discontinuity."""
        g = _make_periodic_grid(101, 2)
        x = g.xs[0].to(torch.float64)
        data = torch.sign(x)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        dx = g.dx[0].item()

        # Far from the discontinuity (|x| > 3*dx), derivative should be ~0
        far_mask = torch.abs(x) > 3 * dx
        sl = _interior_slice(g, margin=6)
        far_mask_int = far_mask[sl]
        derivL_int = derivL[sl]
        derivR_int = derivR[sl]

        max_osc_L = torch.abs(derivL_int[far_mask_int]).max().item()
        max_osc_R = torch.abs(derivR_int[far_mask_int]).max().item()
        # Should be essentially zero away from discontinuity
        assert max_osc_L < 1.0 / dx * 0.1, f"Gibbs oscillation in derivL: {max_osc_L:.2e}"
        assert max_osc_R < 1.0 / dx * 0.1, f"Gibbs oscillation in derivR: {max_osc_R:.2e}"

    def test_kink_function(self):
        """f(x) = |x|: derivative should be +/-1 away from kink."""
        g = _make_periodic_grid(101, 2)
        x = g.xs[0].to(torch.float64)
        data = torch.abs(x)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        dx = g.dx[0].item()
        sl = _interior_slice(g, margin=6)

        # Away from the kink, left deriv should be close to sign(x)
        pos_mask = (x[sl] > 5 * dx)
        neg_mask = (x[sl] < -5 * dx)

        centered = 0.5 * (derivL + derivR)
        centered_int = centered[sl]

        assert torch.allclose(centered_int[pos_mask],
                              torch.ones_like(centered_int[pos_mask]), atol=0.1), \
            "Derivative not ~+1 for x >> 0"
        assert torch.allclose(centered_int[neg_mask],
                              -torch.ones_like(centered_int[neg_mask]), atol=0.1), \
            "Derivative not ~-1 for x << 0"

        # Everything should be finite
        assert torch.isfinite(derivL).all()
        assert torch.isfinite(derivR).all()

    def test_sharp_gradient_bounded(self):
        """tanh(x/eps): derivatives bounded by analytical maximum 1/eps."""
        eps = 0.05
        g = _make_periodic_grid(101, 2)
        x = g.xs[0].to(torch.float64)
        data = torch.tanh(x / eps)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)

        assert torch.isfinite(derivL).all(), "NaN/Inf in derivL for sharp gradient"
        assert torch.isfinite(derivR).all(), "NaN/Inf in derivR for sharp gradient"

        # Analytical max derivative is 1/eps = 20
        max_deriv = max(derivL.abs().max().item(), derivR.abs().max().item())
        # Allow some overshoot but should be bounded
        assert max_deriv < 2.0 / eps, (
            f"Derivative {max_deriv:.2f} exceeds 2/eps={2.0/eps:.2f}"
        )


# ---------------------------------------------------------------------------
# 3. Dimension Tests
# ---------------------------------------------------------------------------

class TestWENO5aDimensions:
    """Verify correct behavior on various grid configurations."""

    def test_1d_grid(self):
        """1D periodic grid: derivative of sin(2*pi*x) matches exact."""
        g = _make_periodic_grid(101, 1)
        data = torch.sin(2 * pi * g.xs[0]).to(torch.float64)
        exact = (2 * pi * torch.cos(2 * pi * g.xs[0])).to(torch.float64)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)

        assert derivL.shape == data.shape
        assert derivR.shape == data.shape
        assert torch.isfinite(derivL).all()
        assert torch.isfinite(derivR).all()

        centered = 0.5 * (derivL + derivR)
        sl = _interior_slice(g, margin=6)
        err = torch.abs(centered[sl] - exact[sl]).max().item()
        assert err < 0.01, f"1D derivative error {err:.2e}"

    def test_3d_grid(self):
        """3D periodic grid: linear function derivative is exact."""
        g = _make_periodic_grid(21, 3)
        # f = 2*x + 3*y + 5*z
        data = (2.0 * g.xs[0] + 3.0 * g.xs[1] + 5.0 * g.xs[2]).to(torch.float64)
        expected_coeffs = [2.0, 3.0, 5.0]

        for dim in range(3):
            derivL, derivR = upwindFirstWENO5a(g, data, dim)
            centered = 0.5 * (derivL + derivR)
            sl = _interior_slice(g, margin=6)
            err = torch.abs(centered[sl] - expected_coeffs[dim]).max().item()
            assert err < 0.05, (
                f"3D dim={dim}: error {err:.2e} for expected coeff {expected_coeffs[dim]}"
            )

    def test_anisotropic_grid(self):
        """Grid with very different dx in each dimension."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = np.array([[101], [21]], dtype=np.int64)
        pdDims = [0, 1]
        g = createGrid(gmin, gmax, N, pdDims, process=True)
        g.xs = [torch.as_tensor(x) for x in g.xs]

        data = (3.0 * g.xs[0] + 7.0 * g.xs[1]).to(torch.float64)

        for dim, expected in [(0, 3.0), (1, 7.0)]:
            derivL, derivR = upwindFirstWENO5a(g, data, dim)
            centered = 0.5 * (derivL + derivR)
            sl = _interior_slice(g, margin=6)
            err = torch.abs(centered[sl] - expected).max().item()
            assert err < 0.1, (
                f"Anisotropic dim={dim}: error {err:.2e}, expected {expected}"
            )

    def test_all_dims_iterated(self):
        """On 3D grid, df/d_dim for f=x+2y+3z matches coefficient."""
        g = _make_periodic_grid(21, 3)
        data = (g.xs[0] + 2.0 * g.xs[1] + 3.0 * g.xs[2]).to(torch.float64)
        coeffs = [1.0, 2.0, 3.0]

        for dim in range(3):
            derivL, derivR = upwindFirstWENO5a(g, data, dim)
            sl = _interior_slice(g, margin=6)
            centered = 0.5 * (derivL + derivR)
            err = torch.abs(centered[sl] - coeffs[dim]).max().item()
            assert err < 0.05, f"dim={dim}: error {err:.2e}, expected {coeffs[dim]}"


# ---------------------------------------------------------------------------
# 4. Edge Case Tests
# ---------------------------------------------------------------------------

class TestWENO5aEdgeCases:
    """Robustness checks for special inputs."""

    def test_constant_function_zero_derivative(self):
        """f = 5.0 everywhere => derivative should be 0."""
        g = _make_periodic_grid(51, 2)
        data = 5.0 * torch.ones_like(g.xs[0], dtype=torch.float64)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        sl = _interior_slice(g, margin=6)
        assert torch.allclose(derivL[sl], torch.zeros_like(derivL[sl]), atol=1e-10), \
            f"derivL not zero: max={derivL[sl].abs().max().item():.2e}"
        assert torch.allclose(derivR[sl], torch.zeros_like(derivR[sl]), atol=1e-10), \
            f"derivR not zero: max={derivR[sl].abs().max().item():.2e}"

    def test_large_magnitude_data(self):
        """Scaling by 1e8 should scale derivative proportionally."""
        g = _make_periodic_grid(51, 2)
        scale = 1e8
        data = scale * torch.sin(pi * g.xs[0]).to(torch.float64)
        exact = scale * pi * torch.cos(pi * g.xs[0]).to(torch.float64)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        centered = 0.5 * (derivL + derivR)
        sl = _interior_slice(g, margin=6)

        assert torch.isfinite(derivL).all(), "NaN/Inf with large-scale data"
        assert torch.isfinite(derivR).all(), "NaN/Inf with large-scale data"

        rel_err = (torch.abs(centered[sl] - exact[sl]) / (torch.abs(exact[sl]) + 1e-30)).max().item()
        assert rel_err < 0.01, f"Large-scale relative error {rel_err:.2e}"

    def test_small_magnitude_data(self):
        """Scaling by 1e-10 should still give correct relative derivative."""
        g = _make_periodic_grid(51, 2)
        scale = 1e-10
        data = scale * torch.sin(pi * g.xs[0]).to(torch.float64)
        exact = scale * pi * torch.cos(pi * g.xs[0]).to(torch.float64)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        centered = 0.5 * (derivL + derivR)
        sl = _interior_slice(g, margin=6)

        assert torch.isfinite(derivL).all(), "NaN/Inf with small-scale data"
        assert torch.isfinite(derivR).all(), "NaN/Inf with small-scale data"

        # Use absolute error scaled by the magnitude
        err = torch.abs(centered[sl] - exact[sl]).max().item()
        assert err < scale * 0.1, f"Small-scale error {err:.2e}"

    def test_generate_all_mode(self):
        """generateAll=True returns lists of length 3, each same shape, all finite."""
        g = _make_periodic_grid(51, 2)
        data = torch.sin(pi * g.xs[0]).to(torch.float64)
        derivL, derivR = upwindFirstWENO5a(g, data, 0, generateAll=True)

        assert isinstance(derivL, list), "derivL should be a list"
        assert isinstance(derivR, list), "derivR should be a list"
        assert len(derivL) == 3, f"Expected 3 left approxs, got {len(derivL)}"
        assert len(derivR) == 3, f"Expected 3 right approxs, got {len(derivR)}"

        for i in range(3):
            assert derivL[i].shape == data.shape, f"derivL[{i}] shape mismatch"
            assert derivR[i].shape == data.shape, f"derivR[{i}] shape mismatch"
            assert torch.isfinite(derivL[i]).all(), f"NaN/Inf in derivL[{i}]"
            assert torch.isfinite(derivR[i]).all(), f"NaN/Inf in derivR[{i}]"


# ---------------------------------------------------------------------------
# 5. Symmetry Tests
# ---------------------------------------------------------------------------

class TestWENO5aSymmetry:
    """Verify mathematical symmetry properties."""

    def test_left_right_symmetry_on_linear(self):
        """For f = ax, derivL and derivR should both equal a on interior."""
        g = _make_periodic_grid(51, 2)
        a = 4.0
        data = (a * g.xs[0]).to(torch.float64)
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        sl = _interior_slice(g, margin=6)

        assert torch.allclose(derivL[sl], torch.full_like(derivL[sl], a), atol=1e-8), \
            f"derivL != {a}: max error {(derivL[sl] - a).abs().max().item():.2e}"
        assert torch.allclose(derivR[sl], torch.full_like(derivR[sl], a), atol=1e-8), \
            f"derivR != {a}: max error {(derivR[sl] - a).abs().max().item():.2e}"

    def test_antisymmetric_function(self):
        """f(x) = x^3 (odd): derivL and derivR should agree on interior."""
        g = _make_periodic_grid(51, 2)
        x = g.xs[0].to(torch.float64)
        data = x ** 3
        derivL, derivR = upwindFirstWENO5a(g, data, 0)
        sl = _interior_slice(g, margin=6)

        # Both should approximate 3x^2
        exact = 3.0 * x[sl] ** 2
        err_L = torch.abs(derivL[sl] - exact).max().item()
        err_R = torch.abs(derivR[sl] - exact).max().item()

        assert err_L < 0.05, f"derivL error on x^3: {err_L:.2e}"
        assert err_R < 0.05, f"derivR error on x^3: {err_R:.2e}"

        # Left and right should be close to each other
        lr_diff = torch.abs(derivL[sl] - derivR[sl]).max().item()
        assert lr_diff < 0.05, f"L-R asymmetry: {lr_diff:.2e}"

    def test_cross_scheme_error_ordering(self):
        """On smooth data: higher-order schemes have smaller left-derivative error.

        We use the left derivative (not centered) because the centered average
        of First-order left+right can coincidentally match ENO2 on symmetric grids.
        """
        g = _make_periodic_grid(41, 2)
        data = (torch.sin(pi * g.xs[0]) * torch.cos(pi * g.xs[1])).to(torch.float64)
        exact = (pi * torch.cos(pi * g.xs[0]) * torch.cos(pi * g.xs[1])).to(torch.float64)
        sl = _interior_slice(g, margin=6)

        schemes = [
            ("First", upwindFirstFirst),
            ("ENO2", upwindFirstENO2),
            ("ENO3a", upwindFirstENO3a),
            ("WENO5a", upwindFirstWENO5a),
        ]
        errors = {}
        for name, scheme in schemes:
            dL, _ = scheme(g, data, 0)
            err = torch.abs(dL[sl] - exact[sl]).max().item()
            errors[name] = err

        # ENO3a should be better than ENO2, and WENO5a better than ENO3a
        assert errors["ENO2"] > errors["ENO3a"], (
            f"ENO2 ({errors['ENO2']:.2e}) should be > ENO3a ({errors['ENO3a']:.2e})"
        )
        assert errors["ENO3a"] > errors["WENO5a"], (
            f"ENO3a ({errors['ENO3a']:.2e}) should be > WENO5a ({errors['WENO5a']:.2e})"
        )
