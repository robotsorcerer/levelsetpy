import itertools
import numpy as np
from chen_mip import priority_matrix, solve_assignment, ChenMIPShield


def test_chen_three_vehicle_all_conflict_reward():
    p = priority_matrix(3)
    u = solve_assignment(p ** 2)
    np.testing.assert_array_equal(u, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert (u * p ** 2).sum() == 77


def test_chen_every_three_vehicle_conflict_pattern_against_enumeration():
    edges = [(i, j) for i in range(3) for j in range(3) if i != j]
    candidates = []
    for bits in itertools.product((0, 1), repeat=6):
        u = np.zeros((3, 3), int)
        for (i, j), bit in zip(edges, bits):
            u[i, j] = bit
        if np.all(u.sum(axis=1) <= 1) and np.all(u + u.T <= 1):
            candidates.append(u)
    for pattern in itertools.product((0, 1), repeat=6):
        rewards = np.full((3, 3), -1)
        for (i, j), active in zip(edges, pattern):
            if active:
                rewards[i, j] = priority_matrix(3)[i, j] ** 2
        u = solve_assignment(rewards)
        assert (rewards * u).sum() == max((rewards * c).sum() for c in candidates)


def test_no_conflict_means_no_avoidance():
    assert not solve_assignment(-np.ones((5, 5))).any()


def test_swept_distance_detects_between_sample_contact():
    from compare_ccbs import swept_dist
    assert swept_dist(np.array([-1., 0.]), np.array([1., 0.])) == 0
    assert swept_dist(np.array([-1., 1.]), np.array([1., 1.])) == 1


def test_fractional_wait_is_not_rounded():
    from compare_ccbs import position
    path = [(0., .375, [0., 0.], [0., 0.]), (.375, 1.375, [0., 0.], [1., 0.])]
    np.testing.assert_allclose(position(path, .25), [0, 0])
    np.testing.assert_allclose(position(path, .5), [.125, 0])
    np.testing.assert_allclose(position(path, 10), [1, 0])


def test_independent_ccbs_checker_rejects_crossing():
    import pytest
    from compare_ccbs import verify_plan
    class Grid:
        def is_free(self, _): return True
    paths = [[(0., 1., [0., 0.], [1., 0.])], [(0., 1., [1., 0.], [0., 0.])]]
    with pytest.raises(AssertionError, match='collision'):
        verify_plan(paths, [(0, 0), (0, 1)], [(0, 1), (0, 0)], Grid())
