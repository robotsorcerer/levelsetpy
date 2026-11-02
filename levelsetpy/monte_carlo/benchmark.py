"""Reproducible manufactured-solution diagnostic; run as python -m benchmark."""
import json
from pathlib import Path
import numpy as np
try:
    from .solver import Coefficient, solve_periodic
except ImportError:  # direct module execution from this project root
    from solver import Coefficient, solve_periodic


def varying_coefficient(t, x):
    return Coefficient(1 + .2*np.cos(x) + .1*np.sin(t),
                       np.full_like(x, .1*np.cos(t)),
                       -.2*np.sin(x), -.2*np.cos(x))


def reference(t, x, delta=.4, speed=.7):
    return np.exp(-.5*delta*t)*np.sin(x-speed*t)


def run():
    results = []
    for mode in ("full", "drop_derivative", "heat"):
        for steps in (100, 200, 400, 800):
            out = solve_periodic(np.sin, varying_coefficient,
                                 lambda t, x, p: .7*p,
                                 steps=steps, mode=mode)
            err = np.max(np.abs(out["value"]-reference(.5, out["x"])))
            grad_ref = np.exp(-.1)*np.cos(out["x"]-.35)
            results.append(dict(mode=mode, steps=steps, points=128,
                                value_sup_error=float(err),
                                gradient_sup_error=float(np.max(np.abs(out["gradient"]-grad_ref)))))
    return dict(problem="periodic linear advection-diffusion, manufactured sine",
                delta=.4, speed=.7, horizon=.5, length=float(2*np.pi),
                coefficient="1 + .2 cos(x) + .1 sin(t)",
                method="Fourier collocation, exponential Euler; no dealiasing",
                scope="deterministic 1D diagnostic; not a reachability benchmark",
                results=results)


if __name__ == "__main__":
    path = Path(__file__).with_name("benchmark_results.json")
    path.write_text(json.dumps(run(), indent=2)+"\n")
    print(path)
