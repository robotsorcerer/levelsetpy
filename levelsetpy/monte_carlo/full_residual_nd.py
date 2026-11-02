"""Multidimensional B.10 integration and independent finite-difference reference.

Elapsed time starts at the initial condition. The transformed equation is
integrated directly with a second-order exponential Runge--Kutta method.
    A fixed collocation source cutoff suppresses high modes; no per-step
    damping filter or positivity clipping is applied. The reference evolves
the original viscous HJ equation with fourth-order centered differences/RK4.
Both use periodic boxes; boundary sensitivity must be measured separately.
"""
from __future__ import annotations

import time
import numpy as np
from scipy import fft


def mesh(bounds, shape):
    axes = tuple(np.linspace(a, b, n, endpoint=False)
                 for (a, b), n in zip(bounds, shape))
    x = tuple(np.meshgrid(*axes, indexing="ij", sparse=True))
    h = tuple((b-a)/n for (a, b), n in zip(bounds, shape))
    waves = []
    for i, (n, hi) in enumerate(zip(shape, h)):
        k = 2*np.pi*(fft.rfftfreq(n, hi) if i == len(shape)-1
                     else fft.fftfreq(n, hi))
        sh = [1]*len(shape); sh[i] = len(k)
        waves.append(k.reshape(sh))
    return axes, x, h, tuple(waves)


def variable_coefficient(bounds, base=.2, amplitude=.04):
    """Smooth positive coefficient with nonzero time and space derivatives."""
    waves = [2*np.pi/(b-a) for a, b in bounds]
    def coefficient(t, x):
        cs = [np.cos(k*q) for k, q in zip(waves, x)]
        product = np.prod(np.broadcast_arrays(*cs), axis=0)
        dc = []
        for i, (k, q) in enumerate(zip(waves, x)):
            d = -amplitude*np.cos(t)*k*np.sin(k*q)
            for j, c in enumerate(cs):
                if j != i: d = d*c
            dc.append(d)
        return (base+amplitude*np.cos(t)*product,
                -amplitude*np.sin(t)*product, tuple(dc),
                -amplitude*np.cos(t)*sum(k*k for k in waves)*product)
    return coefficient


def spectral_derivatives(u, waves):
    uh = fft.rfftn(u)
    multipliers=[]
    for axis,(k,n) in enumerate(zip(waves,u.shape)):
        derivative_k=k.copy()
        if n%2==0:
            index=[slice(None)]*u.ndim
            index[axis]=n//2
            derivative_k[tuple(index)]=0.
        multipliers.append(derivative_k)
    grad = tuple(fft.irfftn(1j*k*uh, s=u.shape) for k in multipliers)
    lap = fft.irfftn(-sum(k*k for k in waves)*uh, s=u.shape)
    return grad, lap


def full_source(t, z, dz, x, coefficient, hamiltonian, delta):
    if not np.all(np.isfinite(z)) or np.min(z) <= 0:
        raise FloatingPointError(f"Transform lost positivity at t={t:.8g}")
    c, ct, dc, lapc = coefficient(t, x)
    if np.min(c) <= 0:
        raise ValueError("Coefficient must be positive")
    v = -np.log(z)/c
    p = tuple(-q/(c*z)+np.log(z)*d/c**2 for q, d in zip(dz, dc))
    alg = c*(hamiltonian(t, x, p)-.5*delta*c*sum(q*q for q in p))
    der = (-ct*v+.5*delta*lapc*v-.5*delta*sum(d*d for d in dc)*v*v
           +delta*(1-c*v)*sum(d*q for d, q in zip(dc, p)))
    return z*(alg+der)


def fd_derivatives(u, h):
    grad = []; lap = np.zeros_like(u)
    for i, hi in enumerate(h):
        p1, m1 = np.roll(u, -1, i), np.roll(u, 1, i)
        p2, m2 = np.roll(u, -2, i), np.roll(u, 2, i)
        grad.append((-p2+8*p1-8*m1+m2)/(12*hi))
        lap += (-p2+16*p1-30*u+16*m1-m2)/(12*hi**2)
    return tuple(grad), lap


def solve(initial, hamiltonian, *, bounds, shape, delta, horizon, steps,
          coefficient=None, kind="full", snapshots=()):
    """Return final field, snapshots and centered-time equation defects.

    Defects are computed independently from centered adjacent-time values
    and spectral spatial derivatives. Initial and final times are excluded
    from the defect, and the reporting mask is supplied by the caller.
    """
    start = time.perf_counter()
    if kind not in {"full", "reference"}: raise ValueError(kind)
    if delta <= 0 or horizon <= 0 or int(steps) != steps or steps < 1:
        raise ValueError("Positive viscosity, horizon and integer steps required")
    if len(bounds) != len(shape) or any(int(n)!=n or n<8 for n in shape):
        raise ValueError("One integer grid size >=8 required per coordinate")
    if any(b<=a for a,b in bounds): raise ValueError("Invalid box")
    axes, x, h, waves = mesh(bounds, shape)
    dt = horizon/steps
    coefficient = coefficient or variable_coefficient(bounds)
    v = np.broadcast_to(initial(x), shape).copy()
    c, *_ = coefficient(0, x)
    u = np.exp(-c*v) if kind == "full" else v.copy()
    lam = -.5*delta*sum(k*k for k in waves)
    source_modes = np.ones(lam.shape, dtype=bool)
    for k, n, hi in zip(waves,shape,h):
        source_modes &= np.abs(k) <= (2*np.pi/(n*hi))*(n//3)
    zarg = dt*lam
    e = np.exp(zarg)
    q1 = np.full_like(lam, dt); q2 = np.full_like(lam, .5*dt)
    np.divide(np.expm1(zarg), lam, out=q1, where=lam != 0)
    np.divide(np.expm1(zarg)-zarg, dt*lam*lam, out=q2, where=lam != 0)
    targets = sorted(set([0., horizon, *map(float, snapshots)]))
    if any(t<0 or t>horizon for t in targets): raise ValueError("Invalid snapshot time")
    indices = {int(round(t/dt)): t for t in targets}
    if len(indices)!=len(targets): raise ValueError("Snapshot times collide")
    if any(abs(i*dt-t)>1e-9 for i, t in indices.items()):
        raise ValueError("Snapshots must align with time steps")
    saved = {0.: v.copy()}
    defects = []
    previous = None
    middle = u.copy()
    # Sample defects at a fixed set of physical times; compare at identical
    # times across refinements, avoiding an initial compatibility transient.
    defect_steps = {int(round(s*steps)) for s in (.25,.5,.75)}
    min_transform = float(np.min(u)) if kind == "full" else None
    for j in range(steps):
        t = j*dt
        if kind == "full":
            def f(at, w):
                grad, _ = spectral_derivatives(w, waves)
                return full_source(at, w, grad, x, coefficient, hamiltonian, delta)
            uh = fft.rfftn(u); f0 = fft.rfftn(f(t, u))*source_modes
            predictor = fft.irfftn(e*uh+q1*f0, s=shape)
            f1 = fft.rfftn(f(t+dt, predictor))*source_modes
            updated = fft.irfftn(e*uh+q1*f0+q2*(f1-f0), s=shape)
            if not np.all(np.isfinite(updated)) or np.min(updated) <= 0:
                raise FloatingPointError(f"Nonpositive/nonfinite transform at step {j+1}")
            min_transform = min(min_transform,float(np.min(updated)))
        else:
            def rhs(at, w):
                p, lap = fd_derivatives(w, h)
                return .5*delta*lap-hamiltonian(at, x, p)
            k1=rhs(t,u);k2=rhs(t+dt/2,u+dt*k1/2)
            k3=rhs(t+dt/2,u+dt*k2/2);k4=rhs(t+dt,u+dt*k3)
            updated=u+dt*(k1+2*k2+2*k3+k4)/6
            if not np.all(np.isfinite(updated)):raise FloatingPointError("Nonfinite reference")
        if j in defect_steps and previous is not None:
            at = j*dt
            grad, lap = spectral_derivatives(middle, waves)
            ut = (updated-previous)/(2*dt)
            if kind == "full":
                c, ct, dc, _ = coefficient(at,x)
                val=-np.log(middle)/c
                p=tuple(-q/(c*middle)+np.log(middle)*d/c**2 for q,d in zip(grad,dc))
                fmid=full_source(at,middle,grad,x,coefficient,hamiltonian,delta)
                transformed_defect=ut-.5*delta*lap-fmid
                # The exact B.10 identity relates the two continuum defects.
                # Independently differentiated recovered v below detects
                # spatial chain-rule/collocation discrepancies as well.
                vp=-np.log(previous)/coefficient(at-dt,x)[0]
                vn=-np.log(updated)/coefficient(at+dt,x)[0]
                pv, lapv=spectral_derivatives(val,waves)
                original_defect=(vn-vp)/(2*dt)+hamiltonian(at,x,pv)-.5*delta*lapv
                chain_gap=transformed_defect+c*middle*original_defect
                defects.append((at, original_defect, transformed_defect, chain_gap))
            else:
                original_defect=ut+hamiltonian(at,x,grad)-.5*delta*lap
                defects.append((at,original_defect,None,None))
        previous=middle;middle=updated;u=updated
        if j+1 in indices:
            saved[indices[j+1]]=(-np.log(u)/coefficient((j+1)*dt,x)[0]
                                 if kind=="full" else u.copy())
    return dict(axes=axes, value=saved[horizon], snapshots=saved, defects=defects,
                steps=steps, dt=dt, shape=shape, bounds=bounds, delta=delta,
                horizon=horizon, kind=kind, min_transform=min_transform,
                runtime=time.perf_counter()-start)
