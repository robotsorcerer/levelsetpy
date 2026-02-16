"""
Tests for ODE CFL integrators: odeCFL1, odeCFL2, odeCFL3.
Exercises basic integration, single-step mode, convergence,
cross-integrator comparisons, and known bugs.
"""

import pytest
import copy
import torch
import numpy as np

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle, DTYPE
from levelsetpy.boundarycondition import addGhostPeriodic
from levelsetpy.spatialderivative import upwindFirstFirst
from levelsetpy.explicitintegration.integration import odeCFL1, odeCFL2, odeCFL3, odeCFLset
from levelsetpy.explicitintegration.dissipation import artificialDissipationGLF
from levelsetpy.explicitintegration.term import termRestrictUpdate, termLaxFriedrichs


def _make_simple_problem():
    """Create a simple 2D eikonal problem for integration tests."""
    gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
    gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
    N = 21 * np.ones((2, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

    data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
    inner_data = Bundle(dict(
        grid=g,
        hamFunc=_simple_hamiltonian,
        partialFunc=_simple_dissipation,
        dissFunc=artificialDissipationGLF,
        CoStateCalc=upwindFirstFirst,
    ))
    scheme_data = Bundle(dict(
        innerFunc=termLaxFriedrichs,
        innerData=inner_data,
        positive=False,
    ))
    return g, data, scheme_data


def _simple_hamiltonian(t, data, derivC, schemeData):
    """Eikonal Hamiltonian: H = |p|."""
    grid = schemeData.grid
    ham = torch.zeros_like(data, dtype=DTYPE)
    for i in range(grid.dim):
        ham = ham + derivC[i] ** 2
    return torch.sqrt(ham)


def _simple_dissipation(t, data, derivMin, derivMax, schemeData, dim):
    """Constant dissipation: alpha = 1."""
    return torch.ones_like(data, dtype=DTYPE)


def _make_options(single_step=False):
    """Create standard options for CFL integrators."""
    opts = Bundle(dict(
        factorCFL=0.5,
        stats='off',
        singleStep='on' if single_step else 'off',
    ))
    return odeCFLset(opts)


class TestOdeCFL1:
    """Tests for the 1st order forward Euler integrator."""

    def test_cfl1_basic_call(self):
        """odeCFL1 runs without crash in single-step mode."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 1.0]
        t, y, _ = odeCFL1(termRestrictUpdate, tspan, y0, _make_options(single_step=True), scheme_data)
        assert isinstance(y, torch.Tensor)

    def test_cfl1_single_step_returns(self):
        """Single step returns before final time."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 1.0]
        t, y, _ = odeCFL1(termRestrictUpdate, tspan, y0, _make_options(single_step=True), scheme_data)
        assert t < 1.0, "Single step should not reach final time"

    def test_cfl1_output_shape(self):
        """Output shape matches input."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 1.0]
        t, y, _ = odeCFL1(termRestrictUpdate, tspan, y0, _make_options(single_step=True), scheme_data)
        assert y.shape == y0.shape

    def test_cfl1_output_finite(self):
        """Output has no NaN/Inf."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 1.0]
        t, y, _ = odeCFL1(termRestrictUpdate, tspan, y0, _make_options(single_step=True), scheme_data)
        assert torch.isfinite(y).all()

    def test_cfl1_solution_changes(self):
        """Solution should differ from initial condition."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.1]
        t, y, _ = odeCFL1(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert not torch.equal(y, y0), "Solution did not change (y never updated)"

    def test_cfl1_time_reaches_target(self):
        """Multi-step integration reaches target time."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.05]
        t, y, _ = odeCFL1(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert abs(t - 0.05) < 1e-10, f"Time did not reach target: {t}"


class TestOdeCFL3:
    """Tests for the 3rd order TVD Runge-Kutta integrator."""

    def test_cfl3_basic_integration(self):
        """odeCFL3 runs and returns (t, y, schemeData)."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.1]
        t, y, sd = odeCFL3(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert isinstance(y, torch.Tensor)
        assert isinstance(t, float)

    def test_cfl3_time_advances(self):
        """Final time reaches target."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.05]
        t, y, _ = odeCFL3(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert abs(t - 0.05) < 1e-10, f"Time did not reach target: {t}"

    def test_cfl3_single_step_mode(self):
        """singleStep='on' exits before final time."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 1.0]
        t, y, _ = odeCFL3(termRestrictUpdate, tspan, y0, _make_options(single_step=True), scheme_data)
        assert t < 1.0, "Single step should not reach final time"

    def test_cfl3_output_shape(self):
        """Output y has same shape as y0."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.1]
        t, y, _ = odeCFL3(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert y.shape == y0.shape

    def test_cfl3_output_finite(self):
        """No NaN/Inf in output."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.1]
        t, y, _ = odeCFL3(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"

    def test_cfl3_solution_changes(self):
        """Solution evolves from initial condition."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        y0_clone = y0.clone()
        tspan = [0, 0.1]
        t, y, _ = odeCFL3(termRestrictUpdate, tspan, y0, _make_options(), scheme_data)
        assert not torch.equal(y, y0_clone), "Solution did not change"

    def test_cfl3_stability_many_steps(self):
        """10 steps without NaN or divergence."""
        _, data, scheme_data = _make_simple_problem()
        y = data.flatten()
        t_now = 0.0
        for step in range(10):
            tspan = [t_now, t_now + 0.02]
            t_now, y, scheme_data = odeCFL3(
                termRestrictUpdate, tspan, y, _make_options(), scheme_data
            )
            assert torch.isfinite(y).all(), f"NaN/Inf at step {step}"
            assert y.abs().max() < 1e6, f"Divergence at step {step}"


class TestCrossIntegrator:
    """Compare odeCFL2 and odeCFL3 behavior."""

    def test_cfl2_cfl3_same_direction(self):
        """Both integrators shrink sphere in the same direction."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.05]

        scheme_data_2 = copy.deepcopy(scheme_data)
        scheme_data_3 = copy.deepcopy(scheme_data)

        _, y2, _ = odeCFL2(termRestrictUpdate, tspan, y0.clone(), _make_options(), scheme_data_2)
        _, y3, _ = odeCFL3(termRestrictUpdate, tspan, y0.clone(), _make_options(), scheme_data_3)

        # Both should evolve in the same general direction (negative mode shrinks)
        diff2 = y2 - y0
        diff3 = y3 - y0
        # Sign correlation: most elements should agree on direction
        agree = (torch.sign(diff2) == torch.sign(diff3)).float().mean()
        assert agree > 0.8, f"CFL2 and CFL3 disagree on direction: {agree:.2f}"

    def test_constant_data_unchanged(self):
        """Constant initial data stays constant under both integrators."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = torch.ones(g.shape, dtype=DTYPE) * 5.0
        inner_data = Bundle(dict(
            grid=g,
            hamFunc=_simple_hamiltonian,
            partialFunc=_simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        scheme_data = Bundle(dict(
            innerFunc=termLaxFriedrichs,
            innerData=inner_data,
            positive=False,
        ))

        y0 = data.flatten()

        sd2 = copy.deepcopy(scheme_data)
        sd3 = copy.deepcopy(scheme_data)

        _, y2, _ = odeCFL2(termRestrictUpdate, [0, 0.1], y0.clone(), _make_options(), sd2)
        _, y3, _ = odeCFL3(termRestrictUpdate, [0, 0.1], y0.clone(), _make_options(), sd3)

        # Constant data should produce zero ydot → no change
        assert torch.allclose(y2, y0, atol=1e-10), f"CFL2 changed constant data: max diff={torch.abs(y2 - y0).max()}"
        assert torch.allclose(y3, y0, atol=1e-10), f"CFL3 changed constant data: max diff={torch.abs(y3 - y0).max()}"

    def test_cfl2_cfl3_deterministic(self):
        """Repeated calls give identical results."""
        _, data, scheme_data = _make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.05]

        sd1 = copy.deepcopy(scheme_data)
        sd2 = copy.deepcopy(scheme_data)

        _, y3a, _ = odeCFL3(termRestrictUpdate, tspan, y0.clone(), _make_options(), sd1)
        _, y3b, _ = odeCFL3(termRestrictUpdate, tspan, y0.clone(), _make_options(), sd2)
        assert torch.equal(y3a, y3b), "Non-deterministic CFL3 results"
