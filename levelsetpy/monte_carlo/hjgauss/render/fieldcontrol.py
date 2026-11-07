"""Vehicle trajectories driven by the gradient of a saved HJ value field.

For each game the optimal rate controls follow from the Hamiltonian used in
hjgauss.validation.cases (evader maximizes, pursuer minimizes). The costate
is p = grad v, taken from a saved snapshot at the nearest elapsed time, with
linear interpolation between snapshots. Both the benchmark (reference) and
HJ-Gauss (full) archives drive their own column.
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load_field(path):
    """Return (value_fn(t, x) -> v, grad_fn(t, x) -> p, times, axes)."""
    d = np.load(path)
    times = np.array(sorted(float(k[2:]) for k in d.files if k.startswith('t_')))
    axes = [d['axis_%d' % i] for i in range(len([k for k in d.files if k.startswith('axis_')]))]
    stack = np.stack([d['t_%g' % t] for t in times])  # (T, *grid)
    dims = len(axes)
    # Gradient on the grid, per snapshot
    grads = []
    for k in range(len(times)):
        g = np.gradient(stack[k], *axes, edge_order=2)  # tuple of dims arrays
        grads.append(np.stack(list(g), 0))
    grads = np.stack(grads)  # (T, dims, *grid)
    pts = (times,) + tuple(axes)
    vi = RegularGridInterpolator(pts, stack, bounds_error=False, fill_value=None)
    gi = [RegularGridInterpolator(pts, grads[:, j], bounds_error=False, fill_value=None)
          for j in range(dims)]

    def value(t, x):
        q = np.c_[np.full(len(x), t), x]
        return vi(q)

    def grad(t, x):
        q = np.c_[np.full(len(x), t), x]
        return np.stack([g(q) for g in gi], -1)

    return value, grad, times, axes


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rockets_controls(x, p, sgn=1.0):
    """Evader ue = sign(p0 x0 - p2); pursuer up = -sign(p1 x0 + p2)."""
    ue = np.sign(sgn * (p[:, 0] * x[:, 0] - p[:, 2]) + 1e-12)
    up = -np.sign(sgn * (p[:, 1] * x[:, 0] + p[:, 2]) + 1e-12)
    return ue, up


def rockets_controls_chase(x, p, sgn=1.0):
    """Evader ue = -sign(p0 x0 - p2); pursuer up = -sign(p1 x0 + p2). Pursuer closes in."""
    ue = -np.sign(p[:, 0] * x[:, 0] - p[:, 2] + 1e-12)
    up = -np.sign(p[:, 1] * x[:, 0] + p[:, 2] + 1e-12)
    return ue, up


def rockets_controls_evade(x, p, sgn=1.0):
    """Evader pushes its range outward (ue = +1); pursuer keeps the chase convention."""
    ue = np.ones(len(x))
    up = -np.sign(p[:, 1] * x[:, 0] + p[:, 2] + 1e-12)
    return ue, up


def rockets_rhs(x, ue, up):
    th = x[:, 2]
    return np.stack([np.cos(th) + ue * x[:, 0],
                     np.sin(th) - 1 + up * x[:, 0],
                     up - ue], -1)


def dubins_controls(x, p, sgn=1.0):
    """Evader we = sign(p0 x1 - p1 x0 - p2); pursuer wp = -sign(p2)."""
    we = np.sign(sgn * (p[:, 0] * x[:, 1] - p[:, 1] * x[:, 0] - p[:, 2]) + 1e-12)
    wp = -np.sign(sgn * p[:, 2] + 1e-12)
    return we, wp


def dubins_controls_swapped(x, p, sgn=1.0):
    """Roles swapped: evader minimizes its switching term, pursuer maximizes it.

    With p = grad v, this makes the pursuer drive toward the evader and the evader away.
    """
    we = -np.sign(sgn * (p[:, 0] * x[:, 1] - p[:, 1] * x[:, 0] - p[:, 2]) + 1e-12)
    wp = -np.sign(sgn * p[:, 2] + 1e-12)   # pursuer keeps the original sign: it chases
    return we, wp


def dubins_rhs(x, we, wp):
    th = x[:, 2]
    return np.stack([-1 + np.cos(th) + we * x[:, 1],
                     np.sin(th) - we * x[:, 0],
                     wp - we], -1)


def rollout(field, x0, T, dt, controls, rhs, sgn=1.0, times_for_field=None, trace=None):
    """RK4 rollout of rate controls chosen from grad v at each stage."""
    value, grad, _, _ = field
    n = int(round(T / dt))
    X = np.empty((n + 1,) + x0.shape); X[0] = x0
    x = x0.copy()
    def f(t, x):
        p = grad(min(t, T), x)
        ue, up = controls(x, p, sgn)
        return rhs(x, ue, up)
    t = 0.0
    for k in range(n):
        if trace is not None:  # record the controls applied on this step, for each agent
            p = grad(min(t, T), x)
            trace.append(controls(x, p, sgn))
        k1 = f(t, x); k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2); k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[:, 2] = wrap(x[:, 2])
        t += dt
        X[k + 1] = x
    return np.linspace(0, T, n + 1), X


if __name__ == '__main__':
    import sys
    f = load_field(sys.argv[1])
    print('times', f[2][:3], 'axes', [a.shape for a in f[3]])
