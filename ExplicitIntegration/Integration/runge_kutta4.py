__all__ = ['dynamics_RK4']

import numpy as onp

def dynamics_RK4(OdeFun, tspan, x, u, v):
    """
    RK4 integrator for a time-invariant dynamical system under a control, u,
    and disturbance, v.

    # See https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

    This function must be called within a loop for a total of N
    steps of integration, Obviously, the smallet the value of T, the better.

    Inputs:
        OdeFun: Right Hand Side of Ode function to be integrated
        tspan: A list [start, end] that specifies over what time horizon to integrate the dynamics
        x: State, must be a list, initial condition
        u: Control, must be a list
        v: Disturbance, must be a list

        Author: Lekan Molu, August 09, 2021
    """
    M = 4 # RK4 steps per interval
    h = 0.2 # time step
    if onp.any(tspan):
        hh = (tspan[1]-tspan[0])/10/M
    X = onp.array(x)
    U = onp.array(u)
    V = onp.array(v)

    for j in range(M):
        if onp.any(tspan): # integrate for this much time steps
            for h in onp.arange(tspan[0], tspan[1], hh):
                k1 = OdeFun(X, U, V)
                k2 = OdeFun(X + h/2 * k1, U, V)
                k3 = OdeFun(X + h/2 * k2, U, V)
                k4 = OdeFun(X + h * k3, U, V)

                X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)
        else:
            k1 = OdeFun(X, U, V)
            k2 = OdeFun(X + h/2 * k1, U, V)
            k3 = OdeFun(X + h/2 * k2, U, V)
            k4 = OdeFun(X + h * k3, U, V)

            X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

    return list(X)
