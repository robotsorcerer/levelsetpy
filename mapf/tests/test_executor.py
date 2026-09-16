import numpy as np
import pytest

from compatible_shield import CompatibleShield
from executor import DubinsParams, init_cpose, rollout
from exp_A_pipeline import run_episode
from mapf_world import Agent, WarehouseGrid


class ConstantNoise:
    def normal(self, *_args):
        return .1


class RecordingNoise:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.draws = []

    def normal(self, *args):
        value = self.rng.normal(*args)
        self.draws.append(value)
        return value


def setup_agents():
    grid = WarehouseGrid()
    agents = [Agent(0, (0, 0), (0, 10)), Agent(1, (4, 0), (4, 10), priority=1)]
    plans = {a.id: ([a.cell, a.goal], []) for a in agents}
    return grid, agents, plans, init_cpose(agents)


def test_velocity_disturbance_scales_with_control_period():
    for substeps in [1, 10, 25]:
        grid, agents, plans, poses = setup_agents()
        params = DubinsParams(substeps=substeps, disturbance_mode="velocity")
        rollout(grid, agents[:1], plans, poses, params, ConstantNoise(), 1)
        # Heading feedback can affect x, but first-order y noise is distinguishable
        # from the old displacement convention even as integration is refined.
        assert 1.05 < poses[0][0] <= 1.101
        assert 0 < poses[0][1] <= .101


def test_one_step_noise_units_are_explicit():
    grid, agents, plans, poses = setup_agents()
    params = DubinsParams(macro_T=.1, substeps=1, disturbance_mode="velocity")
    rollout(grid, agents[:1], plans, poses, params, ConstantNoise(), 1)
    np.testing.assert_allclose(poses[0], [.11, .01, 0.])


@pytest.mark.parametrize("mode,expected", [("velocity", 1), ("legacy_displacement", 0)])
def test_swept_collision_detects_crossing_between_samples(mode, expected):
    grid = WarehouseGrid()
    agents = [Agent(0, (0, 0), (0, 10)), Agent(1, (0, 1), (0, -10), theta=np.pi)]
    plans = {a.id: ([a.cell, a.goal], []) for a in agents}
    poses = init_cpose(agents)
    params = DubinsParams(substeps=1, dist_sigma=0., dist_clip=0., disturbance_mode=mode)
    info = rollout(grid, agents, plans, poses, params, np.random.default_rng(0), 1)
    assert info["collisions"] == expected


def test_contact_is_not_recounted_at_next_replan():
    grid = WarehouseGrid()
    agents = [Agent(0, (0, 0), (0, 0)), Agent(1, (0, 0), (0, 0))]
    plans = {a.id: ([a.cell, a.cell], []) for a in agents}
    poses = init_cpose(agents)
    params = DubinsParams(dist_sigma=0., dist_clip=0., disturbance_mode="velocity")
    for _ in range(2):
        info = rollout(grid, agents, plans, poses, params, np.random.default_rng(0), 1)
        assert info["collisions"] == 0  # Existing contact is not a new onset.


def test_compatible_rollout_preserves_common_random_numbers(table):
    histories = []
    for shield in [None, CompatibleShield(table, fallback="least_violation")]:
        grid, agents, plans, poses = setup_agents()
        noise = RecordingNoise(28)
        params = DubinsParams(disturbance_mode="velocity")
        trace = []
        info = rollout(grid, agents, plans, poses, params, noise, 1, shield=shield, trace=trace)
        assert len(trace) == 10
        assert info["outside_domain_agent_steps"] == 0
        if shield is not None:
            assert all(len(frame["turns"]) == 2 for frame in trace)
        histories.append(noise.draws)
    assert histories[0] == histories[1]
    assert len(histories[0]) == 40


def test_compatible_executor_rejects_mismatched_disturbance_and_bounds(table):
    shield = CompatibleShield(table)
    for params in [DubinsParams(), DubinsParams(disturbance_mode="velocity", dist_clip=.3)]:
        grid, agents, plans, poses = setup_agents()
        with pytest.raises(ValueError):
            rollout(grid, agents, plans, poses, params, np.random.default_rng(0), 1, shield=shield)


def test_pipeline_records_physical_timing_and_safety_limitations(table_path):
    result = run_episode(10, "compatible", brt_path=table_path, n_agents=3,
                         total_ticks=2, exec_steps=2, window=3,
                         params=DubinsParams(disturbance_mode="velocity"),
                         fallback="least_violation")
    assert result["simulated_seconds"] == 4
    assert result["planning_window_seconds"] == 3
    assert result["replanning_period_seconds"] == 2
    assert result["task_progress"] == "committed_discrete_cells"
    assert result["sampled_safety_certified"] is False
    assert result["shield_time_s"] > 0
