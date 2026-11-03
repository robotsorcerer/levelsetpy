"""Fourth-order finite-difference integration of the complete B.10 equation.

The source is evaluated pointwise with all analytic coefficient derivatives.
Classical RK4 integrates both diffusion and source. No spectral source cutoff,
positivity floor, or artificial damping is applied. Independent diagnostic
derivatives are spectral, including the corrected real-FFT Nyquist handling.
"""
from __future__ import annotations
import time
import numpy as np
from hjgauss.core.spectral import mesh,variable_coefficient,fd_derivatives,full_source,spectral_derivatives


def solve(initial,hamiltonian,*,bounds,shape,delta,horizon,steps,coefficient=None,snapshots=()):
    if delta<=0 or horizon<=0 or int(steps)!=steps or steps<1:
        raise ValueError('Positive viscosity, horizon and integer steps required')
    if len(shape)!=len(bounds) or any(int(n)!=n or n<8 for n in shape):
        raise ValueError('One integer grid size >=8 required per coordinate')
    if any(b<=a for a,b in bounds):raise ValueError('Invalid box')
    start=time.perf_counter();dt=horizon/steps
    axes,x,h,waves=mesh(bounds,shape)
    coefficient=coefficient or variable_coefficient(bounds)
    c=coefficient(0,x)[0]
    v=np.broadcast_to(initial(x),shape).copy();u=np.exp(-c*v)
    minimum=float(np.min(u))
    targets=sorted(set([0.,horizon,*map(float,snapshots)]))
    indices={int(round(t/dt)):t for t in targets}
    if any(t<0 or t>horizon for t in targets) or len(indices)!=len(targets) or any(abs(i*dt-t)>1e-9 for i,t in indices.items()):
        raise ValueError('Snapshots must be distinct, in range, and align with steps')
    saved={0.:v.copy()};defects=[];previous=None
    check_steps={int(round(s*steps)) for s in (.25,.5,.75)}
    def rhs(t,z):
        grad,lap=fd_derivatives(z,h)
        return .5*delta*lap+full_source(t,z,grad,x,coefficient,hamiltonian,delta)
    for j in range(steps):
        t=j*dt
        k1=rhs(t,u);k2=rhs(t+dt/2,u+dt*k1/2)
        k3=rhs(t+dt/2,u+dt*k2/2);k4=rhs(t+dt,u+dt*k3)
        updated=u+dt*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(updated)) or np.min(updated)<=0:
            raise FloatingPointError(f'Transform lost positivity at step {j+1}')
        minimum=min(minimum,float(np.min(updated)))
        if j in check_steps and previous is not None:
            c=coefficient(t,x)[0];val=-np.log(u)/c
            vp=-np.log(previous)/coefficient(t-dt,x)[0]
            vn=-np.log(updated)/coefficient(t+dt,x)[0]
            pv,lapv=spectral_derivatives(val,waves)
            original=(vn-vp)/(2*dt)+hamiltonian(t,x,pv)-.5*delta*lapv
            pu,lapu=spectral_derivatives(u,waves)
            transformed=(updated-previous)/(2*dt)-.5*delta*lapu-full_source(t,u,pu,x,coefficient,hamiltonian,delta)
            defects.append((t,original,transformed,transformed+c*u*original))
        previous=u;u=updated
        if j+1 in indices:
            saved[indices[j+1]]=-np.log(u)/coefficient((j+1)*dt,x)[0]
    return dict(axes=axes,value=saved[horizon],snapshots=saved,defects=defects,
                steps=steps,dt=dt,shape=shape,bounds=bounds,delta=delta,horizon=horizon,
                kind='full',method='FD4/RK4 complete transformed equation',
                min_transform=minimum,runtime=time.perf_counter()-start)


def sixth_derivatives(u,h):
    """Independent sixth-order centered first and second differences."""
    grad=[];lap=np.zeros_like(u)
    for axis,hi in enumerate(h):
        p1,m1=np.roll(u,-1,axis),np.roll(u,1,axis)
        p2,m2=np.roll(u,-2,axis),np.roll(u,2,axis)
        p3,m3=np.roll(u,-3,axis),np.roll(u,3,axis)
        grad.append((p3-9*p2+45*p1-45*m1+9*m2-m3)/(60*hi))
        lap+=(2*(p3+m3)-27*(p2+m2)+270*(p1+m1)-490*u)/(180*hi**2)
    return tuple(grad),lap


def original_reference(initial,hamiltonian,*,bounds,shape,delta,horizon,steps,snapshots=()):
    """Independent original-value equation, FD6 in space and RK4 in time."""
    if delta<=0 or horizon<=0 or int(steps)!=steps or steps<1:
        raise ValueError('Positive viscosity, horizon and integer steps required')
    if len(shape)!=len(bounds) or any(int(n)!=n or n<8 for n in shape) or any(b<=a for a,b in bounds):
        raise ValueError('Invalid grid')
    start=time.perf_counter();dt=horizon/steps
    axes,x,h,waves=mesh(bounds,shape);v=np.broadcast_to(initial(x),shape).copy()
    targets=sorted(set([0.,horizon,*map(float,snapshots)]));indices={int(round(t/dt)):t for t in targets}
    if any(t<0 or t>horizon for t in targets) or len(indices)!=len(targets) or any(abs(i*dt-t)>1e-9 for i,t in indices.items()):
        raise ValueError('Invalid snapshot times')
    saved={0.:v.copy()};previous=None;defects=[]
    check_steps={int(round(s*steps)) for s in (.25,.5,.75)}
    def rhs(t,w):
        p,lap=sixth_derivatives(w,h)
        return .5*delta*lap-hamiltonian(t,x,p)
    for j in range(steps):
        t=j*dt;k1=rhs(t,v);k2=rhs(t+dt/2,v+dt*k1/2)
        k3=rhs(t+dt/2,v+dt*k2/2);k4=rhs(t+dt,v+dt*k3)
        updated=v+dt*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(updated)):raise FloatingPointError('Nonfinite FD6 reference')
        if j in check_steps and previous is not None:
            p,lap=spectral_derivatives(v,waves)
            defect=(updated-previous)/(2*dt)+hamiltonian(t,x,p)-.5*delta*lap
            defects.append((t,defect,None,None))
        previous=v;v=updated
        if j+1 in indices:saved[indices[j+1]]=v.copy()
    return dict(axes=axes,value=saved[horizon],snapshots=saved,defects=defects,
                steps=steps,dt=dt,shape=shape,bounds=bounds,delta=delta,horizon=horizon,
                kind='reference',method='FD6/RK4 original viscous equation',
                min_transform=None,runtime=time.perf_counter()-start)
