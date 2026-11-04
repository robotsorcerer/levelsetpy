"""Full generalized Cole--Hopf source; independent NumPy research prototype.

Time is elapsed time since the initial condition, not physical backward time.
The periodic solver is a first-order exponential Euler discretization.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Coefficient:
    value: np.ndarray
    time: np.ndarray
    gradient: np.ndarray
    laplacian: np.ndarray


def _positive(name, value):
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def inverse(omega, c):
    """Recover v without clipping either the transform or coefficient."""
    return -np.log(_positive("omega", omega)) / _positive("c", c)


def inverse_gradient(omega, grad_omega, c, grad_c):
    """One-dimensional gradient, including the derivative of 1/c."""
    omega = _positive("omega", omega)
    c = _positive("c", c)
    return -np.asarray(grad_omega) / (c * omega) + np.log(omega) * grad_c / c**2


def residual(v, p, h, coefficient, delta):
    """Return algebraic and derivative residuals R_alg and R_der."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    c = _positive("c", coefficient.value)
    ct, cx, cxx = coefficient.time, coefficient.gradient, coefficient.laplacian
    alg = c * (h - 0.5 * delta * c * p**2)
    der = (-ct * v + 0.5 * delta * cxx * v
           - 0.5 * delta * cx**2 * v**2
           + delta * (1 - c * v) * cx * p)
    if np.any(~np.isfinite(alg)) or np.any(~np.isfinite(der)):
        raise ValueError("nonfinite residual")
    return alg, der


def source(t, x, omega, grad_omega, coefficient, hamiltonian, delta,
           mode="full"):
    """Evaluate F=omega*(R_alg+R_der) from omega and its gradient.

    'drop_derivative' retains R_alg alone. 'heat' drops both residuals.
    These comparison modes solve different equations.
    """
    v = inverse(omega, coefficient.value)
    p = inverse_gradient(omega, grad_omega, coefficient.value,
                         coefficient.gradient)
    alg, der = residual(v, p, hamiltonian(t, x, p), coefficient, delta)
    if mode == "full":
        return omega * (alg + der)
    if mode == "drop_derivative":
        return omega * alg
    if mode == "heat":
        return np.zeros_like(omega)
    raise ValueError("mode must be full, drop_derivative, or heat")


def solve_periodic(g, coefficient, hamiltonian, *, delta=0.4, horizon=0.5,
                   points=128, steps=1000, length=2*np.pi, mode="full"):
    """Solve on [0,length), returning x, omega, recovered value and gradient.

    coefficient(t,x) supplies c,c_t,c_x,c_xx. No adaptive coefficient
    matching, boundary truncation, obstacles, or stochastic iteration is used.
    Nonpositive iterates raise an error; reduce the step size if needed.
    """
    if points < 8 or int(points) != points or steps < 1 or int(steps) != steps:
        raise ValueError("points >= 8 and steps >= 1 must be integers")
    points, steps = int(points), int(steps)
    if not all(np.isfinite(a) and a > 0 for a in (delta, horizon, length)):
        raise ValueError("delta, horizon, length must be positive and finite")
    x = np.arange(points) * length / points
    k = 2*np.pi*np.fft.fftfreq(points, d=length/points)
    dt = horizon / steps
    lam = -0.5*delta*k**2
    e = np.exp(dt*lam)
    phi = np.full(points, dt)
    np.divide(np.expm1(dt*lam), lam, out=phi, where=lam != 0)
    c0 = coefficient(0., x)
    _positive("c", c0.value)
    omega = _positive("initial omega", np.exp(-c0.value * g(x)))
    for j in range(steps):
        t = j*dt
        grad = np.fft.ifft(1j*k*np.fft.fft(omega)).real
        f = source(t, x, omega, grad, coefficient(t, x), hamiltonian,
                   delta, mode)
        omega = np.fft.ifft(e*np.fft.fft(omega) + phi*np.fft.fft(f)).real
        _positive("omega iterate", omega)
    c = coefficient(horizon, x)
    grad = np.fft.ifft(1j*k*np.fft.fft(omega)).real
    return dict(x=x, omega=omega, value=inverse(omega, c.value),
                gradient=inverse_gradient(omega, grad, c.value, c.gradient))


def gaussian_duhamel(x, t, delta, initial, frozen_source, *, samples=10000,
                     seed=0):
    """Estimate omega and omega_x for a prescribed source on the real line.

    Frozen means supplied independently of these random draws. This is one
    linear Duhamel evaluation, not a complete nonlinear solver. The source
    callback takes arrays (s,y). Importance sampling s=t*(1-u**2) removes
    the terminal-time singularity in the gradient score's second moment
    for bounded source. No positivity/log recovery is performed here.
    """
    if (not all(np.isfinite(a) for a in (x, t, delta)) or t <= 0
            or delta <= 0 or samples < 2 or int(samples) != samples):
        raise ValueError("t, delta > 0 and samples >= 2 required")
    samples = int(samples)
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal(samples)
    y0 = x + np.sqrt(delta*t)*z0
    initial_draws = initial(y0)
    u = 1-rng.random(samples)  # strictly positive, so the score has no 0/0.
    lag = t*u**2
    s = t-lag
    z = rng.standard_normal(samples)
    y = x + np.sqrt(delta*lag)*z
    # 1 / density(s) = 2*sqrt(t*(t-s)) = 2*t*u.
    inv_density = 2*t*u
    f = frozen_source(s, y)
    draws = initial_draws + inv_density*f
    grad_draws = initial_draws*z0/np.sqrt(delta*t) + 2*np.sqrt(t/delta)*z*f
    return dict(omega=float(np.mean(draws)), gradient=float(np.mean(grad_draws)),
                omega_se=float(np.std(draws, ddof=1)/np.sqrt(samples)),
                gradient_se=float(np.std(grad_draws, ddof=1)/np.sqrt(samples)))
