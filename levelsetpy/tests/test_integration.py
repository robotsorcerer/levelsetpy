"""
Integration tests for the LevelSetPy PDE solver pipeline.
Tests: termLaxFriedrichs, termRestrictUpdate, odeCFL2, dissipation functions.
Also includes soak tests, determinism tests, and numerical stability checks.
"""

import pytest
import copy
import time
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle, USE_CUDA, DEVICE, DTYPE, isfield
from levelsetpy.boundarycondition import addGhostPeriodic
from levelsetpy.initialconditions import shapeCylinder, shapeSphere
from levelsetpy.spatialderivative import upwindFirstENO2, upwindFirstFirst
from levelsetpy.explicitintegration.integration import odeCFL2, odeCFLset
from levelsetpy.explicitintegration.dissipation import artificialDissipationGLF, artificialDissipationLLF
from levelsetpy.explicitintegration.term import (
    termRestrictUpdate, termLaxFriedrichs, termSum, termConvection, termNormal,
    termForcing, termDiscount,
)


def make_dubins_grid():
    """Create a small 3D grid suitable for Dubins-type problems."""
    gmin = np.array([[-1.0, -1.0, -pi]]).T
    gmax = np.array([[1.0, 1.0, pi]]).T
    N = 21 * np.ones((3, 1), dtype=np.int64)
    gmax[2, 0] *= (1 - 2 / N[2, 0])
    pdDims = 2  # 3rd dim periodic
    g = createGrid(gmin, gmax, N, pdDims)
    g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]
    return g


def simple_hamiltonian(t, data, derivC, schemeData):
    """A simple test Hamiltonian: H = |p| (eikonal equation)."""
    grid = schemeData.grid
    ham = torch.zeros_like(data, dtype=DTYPE)
    for i in range(grid.dim):
        ham = ham + derivC[i] ** 2
    ham = torch.sqrt(ham)
    return ham


def simple_dissipation(t, data, derivMin, derivMax, schemeData, dim):
    """Simple dissipation: alpha = 1 in all dimensions."""
    return torch.ones_like(data, dtype=DTYPE)


class TestTermLaxFriedrichs:
    """Tests for the Lax-Friedrichs term approximation."""

    def test_basic_call(self):
        """termLaxFriedrichs runs without error on a simple problem."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5
        data = data.to(DTYPE)

        schemeData = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))

        y = data.flatten()
        ydot, stepBound, _ = termLaxFriedrichs(0.0, y, schemeData)

        assert isinstance(ydot, torch.Tensor)
        assert torch.isfinite(ydot).all(), "ydot contains NaN/Inf"
        assert isinstance(stepBound, (int, float))
        assert stepBound > 0, "stepBound must be positive"

    def test_output_shape(self):
        """ydot has correct shape."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (g.xs[0] ** 2 + g.xs[1] ** 2).to(DTYPE)
        schemeData = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        y = data.flatten()
        ydot, _, _ = termLaxFriedrichs(0.0, y, schemeData)
        assert ydot.numel() == y.numel()


class TestTermRestrictUpdate:
    """Tests for the sign-restricted term wrapper."""

    def test_positive_restriction(self):
        """Positive mode: ydot >= 0 everywhere."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        inner_data = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        schemeData = Bundle(dict(
            innerFunc=termLaxFriedrichs,
            innerData=inner_data,
            positive=True,
        ))

        y = data.flatten()
        ydot, stepBound, _ = termRestrictUpdate(0.0, y, schemeData)
        assert (ydot >= -1e-15).all(), "Positive restriction violated"

    def test_negative_restriction(self):
        """Negative mode: ydot <= 0 everywhere."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        inner_data = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        schemeData = Bundle(dict(
            innerFunc=termLaxFriedrichs,
            innerData=inner_data,
            positive=False,
        ))

        y = data.flatten()
        ydot, stepBound, _ = termRestrictUpdate(0.0, y, schemeData)
        assert (ydot <= 1e-15).all(), "Negative restriction violated"


class TestOdeCFL2:
    """Tests for the 2nd order TVD Runge-Kutta ODE integrator."""

    def _make_simple_problem(self):
        """Helper: create a simple 2D eikonal problem."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        inner_data = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        scheme_data = Bundle(dict(
            innerFunc=termLaxFriedrichs,
            innerData=inner_data,
            positive=False,
        ))
        return g, data, scheme_data

    def test_basic_integration(self):
        """odeCFL2 runs and returns valid results."""
        g, data, scheme_data = self._make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.1]
        options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='off'))

        t, y, _ = odeCFL2(termRestrictUpdate, tspan, y0, odeCFLset(options), scheme_data)

        assert isinstance(y, torch.Tensor)
        assert torch.isfinite(y).all(), "NaN/Inf in integration result"
        assert y.shape == y0.shape

    def test_time_advances(self):
        """Final time reaches target."""
        g, data, scheme_data = self._make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.05]
        options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='off'))

        t, y, _ = odeCFL2(termRestrictUpdate, tspan, y0, odeCFLset(options), scheme_data)
        assert abs(t - 0.05) < 1e-10, f"Time did not reach target: {t}"

    def test_single_step_mode(self):
        """singleStep='on' takes exactly one step."""
        g, data, scheme_data = self._make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 1.0]
        options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='on'))

        t, y, _ = odeCFL2(termRestrictUpdate, tspan, y0, odeCFLset(options), scheme_data)
        assert t < 1.0, "Single step should not reach final time for CFL-constrained"

    def test_solution_changes(self):
        """Solution should evolve from initial condition."""
        g, data, scheme_data = self._make_simple_problem()
        y0 = data.flatten()
        tspan = [0, 0.1]
        options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='off'))

        t, y, _ = odeCFL2(termRestrictUpdate, tspan, y0, odeCFLset(options), scheme_data)
        assert not torch.equal(y, y0), "Solution did not change"


class TestDeterminism:
    """Verify deterministic computation across repeated runs."""

    def test_deterministic_integration(self):
        """Same inputs produce bitwise identical outputs."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)

        def run_once():
            inner_data = Bundle(dict(
                grid=g,
                hamFunc=simple_hamiltonian,
                partialFunc=simple_dissipation,
                dissFunc=artificialDissipationGLF,
                CoStateCalc=upwindFirstFirst,
            ))
            sd = Bundle(dict(
                innerFunc=termLaxFriedrichs,
                innerData=inner_data,
                positive=False,
            ))
            y0 = data.flatten().clone()
            tspan = [0, 0.05]
            options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='off'))
            t, y, _ = odeCFL2(termRestrictUpdate, tspan, y0, odeCFLset(options), sd)
            return y

        result1 = run_once()
        result2 = run_once()
        assert torch.equal(result1, result2), "Non-deterministic results detected"


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_no_nan_after_many_steps(self):
        """Integration over many steps doesn't produce NaN."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        inner_data = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        scheme_data = Bundle(dict(
            innerFunc=termLaxFriedrichs,
            innerData=inner_data,
            positive=False,
        ))

        y = data.flatten()
        options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='off'))
        t_now = 0.0
        for step in range(5):
            tspan = [t_now, t_now + 0.05]
            t_now, y, scheme_data = odeCFL2(
                termRestrictUpdate, tspan, y, odeCFLset(options), scheme_data
            )
            assert torch.isfinite(y).all(), f"NaN/Inf at step {step}"
            assert y.abs().max() < 1e10, f"Divergence at step {step}"

    def test_constant_data_stays_constant(self):
        """Constant initial data should produce zero ydot."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = torch.ones(g.shape, dtype=DTYPE) * 5.0
        schemeData = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))

        y = data.flatten()
        ydot, _, _ = termLaxFriedrichs(0.0, y, schemeData)
        # Constant data -> zero derivatives -> zero hamiltonian -> zero ydot
        assert torch.allclose(ydot, torch.zeros_like(ydot), atol=1e-10), \
            f"Constant data produced non-zero ydot: max={ydot.abs().max().item()}"


class TestSoakStability:
    """Soak tests: longer runs checking for drift, leaks, divergence."""

    @pytest.mark.parametrize("N_val", [15, 25])
    def test_soak_multiple_resolutions(self, N_val):
        """Run 10 integration steps at various resolutions."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = N_val * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        inner_data = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        scheme_data = Bundle(dict(
            innerFunc=termLaxFriedrichs,
            innerData=inner_data,
            positive=False,
        ))

        y = data.flatten()
        options = Bundle(dict(factorCFL=0.5, stats='off', singleStep='off'))
        t_now = 0.0
        for step in range(10):
            tspan = [t_now, t_now + 0.02]
            t_now, y, scheme_data = odeCFL2(
                termRestrictUpdate, tspan, y, odeCFLset(options), scheme_data
            )
            assert torch.isfinite(y).all(), f"NaN at step {step}, N={N_val}"
            assert y.abs().max() < 1e6, f"Divergence at step {step}, N={N_val}"


class TestTermSum:
    """Tests for termSum: combines multiple term approximations."""

    def _make_mock_term(self, ydot_val, stepbound_val):
        """Create a mock term function returning fixed values."""
        def term_func(t, y, schemeData):
            ydot = torch.full_like(y, ydot_val, dtype=DTYPE)
            return ydot, stepbound_val, schemeData
        return term_func

    def test_basic_two_terms(self):
        """termSum adds ydot from two independent terms."""
        term1 = self._make_mock_term(1.0, 0.5)
        term2 = self._make_mock_term(2.0, 0.25)

        schemeData = Bundle(dict(
            innerFunc=[term1, term2],
            innerData=[Bundle({}), Bundle({})],
        ))

        y = torch.ones(10, dtype=DTYPE)
        ydot, stepBound, _ = termSum(0.0, y, schemeData)

        assert torch.allclose(ydot, torch.full_like(y, 3.0))
        # stepBound = (1/0.5 + 1/0.25)^-1 = (2 + 4)^-1 = 1/6
        assert abs(stepBound - 1.0 / 6.0) < 1e-10

    def test_single_term_passthrough(self):
        """termSum with one term matches that term exactly."""
        term1 = self._make_mock_term(5.0, 0.1)

        schemeData = Bundle(dict(
            innerFunc=[term1],
            innerData=[Bundle({})],
        ))

        y = torch.ones(10, dtype=DTYPE)
        ydot, stepBound, _ = termSum(0.0, y, schemeData)

        assert torch.allclose(ydot, torch.full_like(y, 5.0))
        assert abs(stepBound - 0.1) < 1e-10

    def test_stepbound_inverse_sum(self):
        """CFL step bound is the inverse sum: (sum 1/sb_i)^-1."""
        sb_values = [0.1, 0.2, 0.5]
        terms = [self._make_mock_term(1.0, sb) for sb in sb_values]
        datas = [Bundle({}) for _ in sb_values]

        schemeData = Bundle(dict(
            innerFunc=terms,
            innerData=datas,
        ))

        y = torch.ones(5, dtype=DTYPE)
        _, stepBound, _ = termSum(0.0, y, schemeData)

        expected = 1.0 / sum(1.0 / sb for sb in sb_values)
        assert abs(stepBound - expected) < 1e-10

    def test_innerdata_updated(self):
        """termSum stores modifications to innerData back."""
        def term_with_counter(t, y, schemeData):
            if not isfield(schemeData, 'count'):
                schemeData.count = 0
            schemeData.count += 1
            ydot = torch.zeros_like(y, dtype=DTYPE)
            return ydot, 1.0, schemeData

        d1, d2 = Bundle({}), Bundle({})
        schemeData = Bundle(dict(
            innerFunc=[term_with_counter, term_with_counter],
            innerData=[d1, d2],
        ))

        y = torch.ones(5, dtype=DTYPE)
        _, _, sd_out = termSum(0.0, y, schemeData)

        assert sd_out.innerData[0].count == 1
        assert sd_out.innerData[1].count == 1

    def test_with_real_lax_friedrichs(self):
        """termSum with two real LF terms produces valid output."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)

        inner_data1 = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))
        inner_data2 = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        ))

        schemeData = Bundle(dict(
            innerFunc=[termLaxFriedrichs, termLaxFriedrichs],
            innerData=[inner_data1, inner_data2],
        ))

        y = data.flatten()
        ydot, stepBound, _ = termSum(0.0, y, schemeData)

        assert isinstance(ydot, torch.Tensor)
        assert torch.isfinite(ydot).all()
        assert stepBound > 0

        # Sum of two identical terms should be 2x the single term
        ydot_single, sb_single, _ = termLaxFriedrichs(0.0, y, Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationGLF,
            CoStateCalc=upwindFirstFirst,
        )))
        assert torch.allclose(ydot, 2 * ydot_single, atol=1e-10)
        expected_sb = 1.0 / (2.0 / sb_single)
        assert abs(stepBound - expected_sb) < 1e-10


class TestArtificialDissipationLLF:
    """Tests for Local Lax-Friedrichs dissipation."""

    def _make_problem(self):
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        schemeData = Bundle(dict(
            grid=g,
            hamFunc=simple_hamiltonian,
            partialFunc=simple_dissipation,
            dissFunc=artificialDissipationLLF,
            CoStateCalc=upwindFirstFirst,
        ))
        return g, data, schemeData

    def test_basic_call(self):
        """LLF dissipation runs without error."""
        g, data, schemeData = self._make_problem()
        derivL = []
        derivR = []
        for dim in range(g.dim):
            dL, dR = upwindFirstFirst(g, data, dim)
            derivL.append(dL)
            derivR.append(dR)

        diss, stepBound = artificialDissipationLLF(0.0, data, derivL, derivR, schemeData)
        assert isinstance(diss, torch.Tensor)
        assert torch.isfinite(diss).all()
        assert stepBound > 0

    def test_output_shape(self):
        """Dissipation has same shape as input data."""
        g, data, schemeData = self._make_problem()
        derivL, derivR = [], []
        for dim in range(g.dim):
            dL, dR = upwindFirstFirst(g, data, dim)
            derivL.append(dL)
            derivR.append(dR)

        diss, _ = artificialDissipationLLF(0.0, data, derivL, derivR, schemeData)
        assert diss.shape == data.shape

    def test_less_dissipation_than_glf(self):
        """LLF should produce same or less dissipation than GLF."""
        g, data, schemeData = self._make_problem()
        derivL, derivR = [], []
        for dim in range(g.dim):
            dL, dR = upwindFirstFirst(g, data, dim)
            derivL.append(dL)
            derivR.append(dR)

        diss_llf, _ = artificialDissipationLLF(0.0, data, derivL, derivR, schemeData)
        diss_glf, _ = artificialDissipationGLF(0.0, data, derivL, derivR, schemeData)

        # LLF dissipation magnitude should be <= GLF at every point
        assert (torch.abs(diss_llf) <= torch.abs(diss_glf) + 1e-10).all(), \
            "LLF dissipation exceeds GLF dissipation"

    def test_constant_data_zero_dissipation(self):
        """Constant data should produce zero dissipation."""
        g, _, schemeData = self._make_problem()
        data = torch.ones(g.shape, dtype=DTYPE) * 5.0
        derivL, derivR = [], []
        for dim in range(g.dim):
            dL, dR = upwindFirstFirst(g, data, dim)
            derivL.append(dL)
            derivR.append(dR)

        diss, _ = artificialDissipationLLF(0.0, data, derivL, derivR, schemeData)
        assert torch.allclose(diss, torch.zeros_like(diss), atol=1e-10)


class TestTermConvection:
    """Tests for convective term approximation."""

    def test_basic_call(self):
        """termConvection runs on a simple constant velocity problem."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)

        # Constant velocity field: v = (1, 0) — motion in x direction
        velocity = [torch.ones(g.shape, dtype=DTYPE), torch.zeros(g.shape, dtype=DTYPE)]

        schemeData = Bundle(dict(
            grid=g,
            derivFunc=upwindFirstFirst,
            velocity=velocity,
        ))

        y = data.flatten()
        ydot, stepBound, _ = termConvection(0.0, y, schemeData)

        assert isinstance(ydot, torch.Tensor)
        assert torch.isfinite(ydot).all()
        assert stepBound > 0

    def test_output_shape(self):
        """ydot shape matches y shape."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (g.xs[0] + g.xs[1]).to(DTYPE)
        velocity = [torch.ones(g.shape, dtype=DTYPE), torch.zeros(g.shape, dtype=DTYPE)]

        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, velocity=velocity))
        y = data.flatten()
        ydot, _, _ = termConvection(0.0, y, schemeData)
        assert ydot.numel() == y.numel()

    def test_zero_velocity_zero_ydot(self):
        """Zero velocity should produce zero ydot."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        velocity = [torch.zeros(g.shape, dtype=DTYPE), torch.zeros(g.shape, dtype=DTYPE)]

        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, velocity=velocity))
        y = data.flatten()
        ydot, _, _ = termConvection(0.0, y, schemeData)
        assert torch.allclose(ydot, torch.zeros_like(ydot), atol=1e-10)

    def test_constant_data_zero_ydot(self):
        """Constant data should produce zero ydot regardless of velocity."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = torch.ones(g.shape, dtype=DTYPE) * 3.0
        velocity = [torch.ones(g.shape, dtype=DTYPE) * 2.0,
                     torch.ones(g.shape, dtype=DTYPE) * -1.0]

        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, velocity=velocity))
        y = data.flatten()
        ydot, _, _ = termConvection(0.0, y, schemeData)
        assert torch.allclose(ydot, torch.zeros_like(ydot), atol=1e-10)


class TestTermNormal:
    """Tests for normal direction motion term."""

    def test_basic_call_scalar_speed(self):
        """termNormal runs with scalar speed."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)

        schemeData = Bundle(dict(
            grid=g,
            derivFunc=upwindFirstFirst,
            speed=1.0,
        ))

        y = data.flatten()
        ydot, stepBound, _ = termNormal(0.0, y, schemeData)

        assert isinstance(ydot, torch.Tensor)
        assert torch.isfinite(ydot).all()
        assert stepBound > 0

    def test_output_shape(self):
        """ydot shape matches y shape."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, speed=1.0))
        y = data.flatten()
        ydot, _, _ = termNormal(0.0, y, schemeData)
        assert ydot.numel() == y.numel()

    def test_array_speed(self):
        """termNormal works with array-valued speed."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        speed = torch.ones(g.shape, dtype=DTYPE) * 0.5

        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, speed=speed))
        y = data.flatten()
        ydot, stepBound, _ = termNormal(0.0, y, schemeData)

        assert torch.isfinite(ydot).all()
        assert stepBound > 0

    def test_zero_speed_zero_ydot(self):
        """Zero speed should produce zero ydot."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, speed=0.0))
        y = data.flatten()
        ydot, _, _ = termNormal(0.0, y, schemeData)
        assert torch.allclose(ydot, torch.zeros_like(ydot), atol=1e-10)

    def test_positive_speed_shrinks_sphere(self):
        """Positive speed should shrink a sphere (signed distance function)."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        data = (torch.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2) - 0.5).to(DTYPE)
        schemeData = Bundle(dict(grid=g, derivFunc=upwindFirstFirst, speed=1.0))
        y = data.flatten()
        ydot, _, _ = termNormal(0.0, y, schemeData)

        # For speed > 0, the level set moves inward (phi increases),
        # so ydot = -speed * |grad phi| should be negative (since -speed < 0).
        # Interior points (data < 0) should see phi increasing (positive ydot)
        # Actually: ydot = -a * |grad phi|. For a=1, ydot <= 0 everywhere.
        # This means the sphere shrinks (phi gets more positive = surface moves inward).
        data_reshaped = data
        interior = (data_reshaped < -0.1).flatten()
        if interior.any():
            # Interior ydot should be negative (phi increasing = shrinking)
            assert (ydot[interior] <= 1e-10).all()


class TestTermForcing:
    """Tests for the forcing term (no spatial derivative)."""

    def test_scalar_forcing(self):
        """Scalar forcing produces uniform ydot."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)

        schemeData = Bundle(dict(grid=g, forcing=3.0))
        y = torch.ones(g.shape, dtype=DTYPE).flatten()
        ydot, stepBound, _ = termForcing(0.0, y, schemeData)

        assert ydot.numel() == y.numel()
        # ydot = -forcing = -3.0 everywhere
        assert torch.allclose(ydot, torch.full_like(ydot, -3.0))
        assert stepBound == float('inf')

    def test_array_forcing(self):
        """Array forcing produces spatially varying ydot."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)
        g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]

        forcing = g.xs[0].to(DTYPE)  # spatially varying
        schemeData = Bundle(dict(grid=g, forcing=forcing))
        y = torch.ones(g.shape, dtype=DTYPE).flatten()
        ydot, stepBound, _ = termForcing(0.0, y, schemeData)

        assert torch.isfinite(ydot).all()
        assert stepBound == float('inf')

    def test_callable_forcing(self):
        """Function handle forcing works."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)

        def forcing_func(t, data, sd):
            return torch.ones_like(data, dtype=DTYPE) * 2.0

        schemeData = Bundle(dict(grid=g, forcing=forcing_func))
        y = torch.ones(g.shape, dtype=DTYPE).flatten()
        ydot, stepBound, _ = termForcing(0.0, y, schemeData)

        assert torch.allclose(ydot, torch.full_like(ydot, -2.0))
        assert stepBound == float('inf')


class TestTermDiscount:
    """Tests for the discount term (lambda * phi)."""

    def test_scalar_discount(self):
        """Scalar discount factor works."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)

        schemeData = Bundle(dict(grid=g, lambder=0.5))
        y = torch.ones(g.shape, dtype=DTYPE).flatten() * 4.0
        ydot, stepBound, _ = termDiscount(0.0, y, schemeData)

        assert ydot.numel() == y.numel()
        # ydot = -(lambda * data) = -(0.5 * 4.0) = -2.0
        assert torch.allclose(ydot, torch.full_like(ydot, -2.0))
        assert stepBound == float('inf')

    def test_zero_data_zero_ydot(self):
        """Zero data should give zero ydot regardless of discount."""
        gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
        gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
        N = 21 * np.ones((2, 1), dtype=np.int64)
        g = createGrid(gmin, gmax, N, process=True)

        schemeData = Bundle(dict(grid=g, lambder=10.0))
        y = torch.zeros(g.shape, dtype=DTYPE).flatten()
        ydot, _, _ = termDiscount(0.0, y, schemeData)

        assert torch.allclose(ydot, torch.zeros_like(ydot))
