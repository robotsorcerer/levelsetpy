import numpy as np

def dynamics_RK4(OdeFun, tspan, x, u, v):
    """
    # RK4 integrator for a time-invariant dynamical system under a control, u,
    and disturbance, v.

    # See https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

    This impl adopted from unstable-zeros's learning CBFs example for two airplanes

    https://github.com/unstable-zeros/learning-cbfs/blob/master/airplane_example/learning_cbfs_airplane.ipynb

    This function must be called within a loop for a total of N
    steps of integration, Obviously, the smallet the value of T, the better

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
    if np.any(tspan):
        hh = (tspan[1]-tspan[0])/10/M
    X = np.array(x)
    U = np.array(u)
    V = np.array(v)

    for j in range(M):
        if np.any(tspan): # integrate for this much time steps
            for h in np.arange(tspan[0], tspan[1], hh):
                k1 = OdeFun(None, X, U, V)
                k2 = OdeFun(None, X + h/2 * k1, U, V)
                k3 = OdeFun(None, X + h/2 * k2, U, V)
                k4 = OdeFun(None, X + h * k3, U, V)

                X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)
        else:
            k1 = OdeFun(None, X, U, V)
            k2 = OdeFun(None, X + h/2 * k1, U, V)
            k3 = OdeFun(None, X + h/2 * k2, U, V)
            k4 = OdeFun(None, X + h * k3, U, V)

            X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

    return list(X)
