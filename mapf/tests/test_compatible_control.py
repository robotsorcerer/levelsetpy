import itertools

import numpy as np
import pytest

from compatible_brt import ReachabilityTable, relative_dynamics, solve_brt
from compatible_shield import (CompatibleShield, InfeasibleSafetyConstraints,
                               pairwise_affine, project_interval)
from hj_conflict import relative_state, wrap_angle


def test_relative_dynamics_matches_world_derivative():
    rng = np.random.default_rng(8)
    for _ in range(30):
        poses = rng.normal(size=(2, 3))
        turns = rng.uniform(-1, 1, 2)
        speeds = rng.uniform(0, 1, 2)
        q = relative_state(*poses)
        velocity = np.column_stack((speeds * np.cos(poses[:, 2]),
                                    speeds * np.sin(poses[:, 2]), turns))
        eps = 1e-7
        difference = relative_state(*(poses + eps * velocity)) - q
        difference[2] = wrap_angle(difference[2])
        np.testing.assert_allclose(difference / eps,
                                   relative_dynamics(q, *turns, *speeds), atol=2e-6)


def test_affine_disturbance_bound_matches_corner_enumeration():
    q = np.array([1.1, -.6, 1.8])
    gradient = np.array([.7, -.2, .4])
    heading, ego_turn, disturbance = .73, -.2, .12
    b, a = pairwise_affine(q, gradient, heading, .8, .4, 1., disturbance)
    rotation = np.array([[np.cos(heading), np.sin(heading)],
                         [-np.sin(heading), np.cos(heading)]])
    values = []
    for other_turn in [-1., 1.]:
        for corners in itertools.product([-disturbance, disturbance], repeat=4):
            error = rotation @ (np.array(corners[2:]) - np.array(corners[:2]))
            derivative = relative_dynamics(q, ego_turn, other_turn, .8, .4)
            derivative[:2] += error
            values.append(gradient @ derivative)
    assert b * ego_turn + a == pytest.approx(min(values))


def test_interval_preserves_all_compatible_constraints():
    # One threat permits [.2,1], another [-1,.6]. Both require one shared turn.
    result = project_interval(1., [1., -1.], [-.2, .6], 1.)
    assert result.feasible
    assert result.lower == pytest.approx(.2)
    assert result.upper == pytest.approx(.6)
    assert result.control == pytest.approx(.6)
    assert result.min_residual >= -1e-12


@pytest.mark.parametrize("slopes,offsets", [([1., -1.], [-.7, -.7]), ([0.], [-1.])])
def test_infeasibility_is_reported_and_fallback_is_minimax(slopes, offsets):
    result = project_interval(.3, slopes, offsets, 1.)
    assert not result.feasible
    assert -1 <= result.control <= 1
    candidates = np.linspace(-1, 1, 20001)
    best_grid = np.min(np.array(slopes)[:, None] * candidates + np.array(offsets)[:, None], axis=0).max()
    assert result.min_residual >= best_grid - 1e-8
    assert result.min_residual < 0


def test_interval_randomized_against_independent_linear_program():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(17)
    for _ in range(80):
        b, c = rng.normal(size=(2, 7))
        result = project_interval(rng.normal(), b, c, 1.)
        oracle = scipy.linprog([0.], A_ub=-b[:, None], b_ub=c, bounds=[(-1., 1.)], method="highs")
        assert result.feasible == oracle.success
        if not result.feasible:
            minimax = scipy.linprog([0., -1.], A_ub=np.column_stack((-b, np.ones_like(b))),
                                    b_ub=c, bounds=[(-1., 1.), (None, None)], method="highs")
            assert result.min_residual == pytest.approx(minimax.x[1], abs=1e-9)


def test_table_gradient_matches_value_finite_difference(table):
    q = np.random.default_rng(2).uniform([-3.5, -3.5, -3.], [3.5, 3.5, 3.], (50, 3))
    value, gradient = table.value_and_gradient(q)
    assert value.shape == (50,)
    for dim in range(3):
        delta = np.zeros(3)
        delta[dim] = 1e-6
        finite_difference = (table.value(q + delta) - table.value(q - delta)) / 2e-6
        np.testing.assert_allclose(gradient[:, dim], finite_difference, atol=1e-8)


def test_periodic_seam_and_all_limiting_gradients(table):
    q = np.array([1.13, .73, -np.pi])
    equivalent = q + np.array([0., 0., 2 * np.pi])
    for actual, expected in zip(table.value_and_gradient(q), table.value_and_gradient(equivalent)):
        np.testing.assert_allclose(actual, expected)
    branches = table.gradient_branches(q)
    for sign in [-1, 1]:
        near = q + np.array([0., 0., sign * 1e-8])
        _, gradient = table.value_and_gradient(near)
        assert np.min(np.linalg.norm(branches - gradient, axis=1)) < 1e-6


def test_table_geometric_enclosure(table):
    q = np.random.default_rng(3).uniform([-4, -4, -np.pi], [4, 4, np.pi], (10000, 3))
    h = table.value(q) - table.geometric_padding - table.margin
    physical = np.linalg.norm(q[:, :2], axis=1) - table.capture_radius
    assert np.all(h <= physical + 1e-12)
    assert np.all(h[physical < 0] < 0)


def test_limiting_gradients_on_spatial_faces_include_both_sides(table):
    for dim in [0, 1]:
        for node in table.axes[dim][1:-1]:
            q = np.array([1.13, .73, .37])
            q[dim] = node
            branches = table.gradient_branches(q)
            for sign in [-1, 1]:
                near = q.copy()
                near[dim] += sign * 1e-8
                _, gradient = table.value_and_gradient(near)
                assert np.min(np.linalg.norm(branches - gradient, axis=1)) < 1e-6


def test_legacy_and_out_of_domain_cache_queries_rejected(tmp_path, table):
    path = tmp_path / "legacy.npz"
    np.savez(path, V=np.zeros((2, 2, 2)), meta=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="Legacy"):
        ReachabilityTable(path)
    with pytest.raises(ValueError, match="outside"):
        table.value([10., 0., 0.])


def test_positive_nodal_overshoot_is_rejected(tmp_path, table_path):
    with np.load(table_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    geometric = np.hypot(arrays["x1_axis"][:, None], arrays["x2_axis"][None, :]) - .5
    arrays["V"] = np.broadcast_to(geometric[..., None], arrays["V"].shape).copy() + 1e-12
    path = tmp_path / "overshoot.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="nodal"):
        ReachabilityTable(path)


def compatible_three_robot_poses():
    return np.array([[0., 0., 0.],
                     [1.296183155848269, -.3140184638331167, -1.6311606000133607],
                     [1.229521980823014, .7950187272262399, -.47623819441489124]])


def test_actual_three_robot_state_supports_two_simultaneous_constraints(table):
    poses = compatible_three_robot_poses()
    all_pairs = CompatibleShield(table, fallback="least_violation")
    one_pair = CompatibleShield(table, fallback="least_violation", mode="most_critical")
    result = all_pairs.filter(poses, np.ones(3), np.ones(3), dt=.1)
    single = one_pair.filter(poses, np.ones(3), np.ones(3), dt=.1)
    assert result.active_pairs[0] == 2
    assert result.min_barrier[0] > 0
    assert result.interval_feasible[0]
    assert result.all_constraints_satisfied[0]
    assert result.lower[0] > 0
    assert result.upper[0] < 1
    assert not single.all_constraints_satisfied[0]
    assert single.min_residual[0] < -.01


def test_fleet_permutation_does_not_change_compatible_commands(table):
    poses = compatible_three_robot_poses()
    nominal = np.array([1., -.7, .1])
    shield = CompatibleShield(table, fallback="least_violation")
    base = shield.filter(poses, nominal, np.ones(3), dt=.1)
    for order in itertools.permutations(range(3)):
        idx = np.array(order)
        permuted = shield.filter(poses[idx], nominal[idx], np.ones(3), dt=.1)
        np.testing.assert_allclose(permuted.turns, base.turns[idx], atol=1e-10)
        np.testing.assert_equal(permuted.all_constraints_satisfied, base.all_constraints_satisfied[idx])


def test_unsafe_initial_state_raises_in_strict_mode(table):
    shield = CompatibleShield(table)
    with pytest.raises(InfeasibleSafetyConstraints) as error:
        shield.filter([[0., 0., 0.], [.1, 0., np.pi]], [0., 0.], [1., 1.], dt=.1)
    assert error.value.decision.needs_replan.any()


def test_far_pair_is_screened_with_a_physical_distance_bound(table):
    result = CompatibleShield(table).filter([[0, 0, 0], [20, 0, np.pi]], [0., 0.], [1., 1.], dt=.1)
    assert result.far_pairs == 2
    assert not result.needs_replan.any()


def test_solver_monotonicity_and_heading_symmetry(table):
    x, y, theta = np.meshgrid(*table.axes, indexing="ij")
    assert np.all(table.V <= np.hypot(x, y) - table.capture_radius + 1e-12)
    reflected_heading = (-np.arange(len(table.axes[2]))) % len(table.axes[2])
    np.testing.assert_allclose(table.V, table.V[:, ::-1, :][:, :, reflected_heading], atol=1e-9)


def test_three_second_demonstration_keeps_all_shared_constraints_feasible(tmp_path):
    from compatible_control_demo import demonstrate
    path = tmp_path / "demo_table.npz"
    solve_brt(path, n_xy=81, n_theta=64, relative_disturbance=2 * np.sqrt(2) * .12)
    table = ReachabilityTable(path, margin=.1)
    result = demonstrate(table)
    all_pairs, one_pair = result["modes"]["all"], result["modes"]["most_critical"]
    assert min(all_pairs["initial_min_barrier"]) > 0
    assert all_pairs["initial_active_pairs"][0] == 2
    assert all_pairs["violated_agent_steps"] == 0
    assert all_pairs["compatible_multi_threat_steps"] > 0
    assert one_pair["violated_agent_steps"] > 0
    for mode in result["modes"].values():
        assert mode["min_separation"] > table.capture_radius
        assert mode["min_barrier"] > 0
