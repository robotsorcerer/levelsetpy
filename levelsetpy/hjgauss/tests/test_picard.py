"""Regression tests for the nonlinear nested Picard evaluator."""
import numpy as np

from picard import NestedPicardSolver, constant_coefficient
from scaling import analytic_problem, run_case


def solver(dimension=2, samples=64):
    initial, hamiltonian, _ = analytic_problem(dimension)
    return NestedPicardSolver(
        initial,
        constant_coefficient(0.7),
        hamiltonian,
        delta=0.4,
        samples=samples,
    )


def test_reproducible_nested_evaluation():
    first = solver(samples=8).evaluate(1, 0.1, np.zeros(2), seed=17)
    second = solver(samples=8).evaluate(1, 0.1, np.zeros(2), seed=17)
    assert first.value == second.value
    np.testing.assert_array_equal(first.gradient, second.gradient)
    assert first.stats == second.stats


def test_level_zero_matches_heat_solution_with_sampling_tolerance():
    dimension = 2
    _, _, truth = analytic_problem(dimension)
    result = solver(dimension, samples=100000).evaluate(0, 0.15, np.zeros(dimension), seed=9)
    expected, expected_gradient = truth(0.15, np.zeros(dimension))
    assert abs(result.value - expected) < 2e-3
    assert np.linalg.norm(result.gradient - expected_gradient) < 8e-3


def test_nonlinear_levels_are_positive_and_count_source_work():
    result = solver(samples=6).evaluate(2, 0.1, np.zeros(2), seed=31)
    assert result.transformed_value > 0
    assert result.stats["source_evaluations"] == 6 + 6 * 6
    assert result.stats["maximum_recursion_depth"] == 2


def test_dimension_payload_bound_is_linear_at_fixed_configuration():
    low = run_case(2, 4, 1, 4, 0.05)
    high = run_case(8, 4, 1, 4, 0.05)
    assert high["depth_linear_coordinate_bound_bytes"] == 4 * low["depth_linear_coordinate_bound_bytes"]
    assert np.isfinite(high["runtime_seconds"])


def test_invalid_inputs_are_rejected():
    try:
        solver(samples=3)
    except ValueError:
        pass
    else:
        raise AssertionError("odd sample counts must be rejected")
