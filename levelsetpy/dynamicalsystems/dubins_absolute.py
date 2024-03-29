__all__ = ["DubinsVehicleAbs"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"
__date__        = "Dec. 21, 2021"
__comment__     = "Two Dubins Vehicle in Absolute Coordinates"

import time
import cupy as cp
import numpy as np
from levelsetpy.utilities import eps

class DubinsVehicleAbs():
    def __init__(self, grid, u_bound=+5, w_bound=+5, \
                 init_state=[0,0,0], rw_cov=0.0, \
                 axis_align=2, center=None, label=None,
                 neigh_rad=.4):
        """
            Dubins Vehicle Dynamics in absolute coordinates.
            Please consult Merz, 1972 for a detailed reference.

            Dynamics:
            ==========
                \dot{x}_1 = v cos x_3
                \dot{x}_2 = v sin x_3
                \dot{x}_3 = w

            Parameters:
            ===========
                grid: an np.meshgrid state space on which we are
                resolving this vehicular dynamics. This grid does not have
                a value function (yet!) until it's part of a flock
                u_bound: absolute value of the linear speed of the vehicle.
                w_bound: absolute value of the angular speed of the vehicle.
                init_state: initial position and orientation of a bird on the grid
                rw_cov: random covariance scalar for initiating the stochasticity
                on the grid.
                center: location of this bird's value function on the grid
                axis_align: periodic dimension on the grid to be created
                neigh_rad: neighboring radius that defines the circle where nearest neighbors are counted.
        """

        assert label is not None, "label of an agent cannot be empty"

        self.grid        = grid
        # self.v = lambda u: u*u_bound
        # self.w = lambda w: w*w_bound
        self.v = u_bound
        self.w = w_bound
        self.neigh_rad = neigh_rad

        # this is a vector defined in the direction of its nearest neighbor
        self.u = None
        self.deltaT = eps # use system eps for a rough small start due to in deltaT
        self.rand_walk_cov = rw_cov

        self.center = center
        self.axis_align = axis_align

        if not np.any(init_state):
            init_state = np.zeros((grid.shape))

        # position this bird at in the state space
        self.initialize(init_state, init_random)

    def initialize(self, init_state, init_random):
        """
            simulate each agent's position in a flock as a random walk
            Parameters
            ==========
            .init_state: current state of a bird in the state space
                (does not have to be an initial state/could be a current
                state during simulation).
        """
        if init_random:
            # time between iterations
            W = np.asarray(([self.deltaT**2/2])).T*np.identity(init_state.shape[-1])
            WWT = W@W.T*self.rand_walk_cov**2
            WWCov = np.tile(WWT, [len(init_state), 1, 1])
            rand_walker = init_state*WWCov

            self.state = init_state + rand_walker
        else:
            self.state = init_state

        return self.state

    def dynamics(self, cur_state):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x}_1 = v cos x_3
            \dot{x}_2 = v sin x_3
            \dot{x}_3 = w * I[sizeof(x_3)]
        """
        if not np.any(cur_state):
            cur_state = self.grid.xs

        xdot = [
                self.v * np.cos(cur_state[2]),
                self.v * np.sin(cur_state[2]),
                self.w * np.ones_like(cur_state[2])
        ]
        return np.asarray(xdot)

    def update_values(self, cur_state, t_span=None):
        """
            Birds use an optimization scheme to keep
            separated distances from one another.

            'even though vision is the main mechanism of interaction,
            optimization determines the anisotropy of neighbors, and
            not the eye's structure. There is also the possibility that
            each individual keeps the front neighbor at larger distances
            to avoid collisions. This collision avoidance mechanism is
            vision-based but not related to the eye's structure.'

            Parameters
            ==========
            cur_state: position and orientation.
                i.e. [x1, x2, θ] at this current position
            t_span: time_span as a list [t0, tf] where
                .t0: initial integration time
                .tf: final integration time
        """
        assert not np.any(cur_state), "current state cannot be empty."

        M, h = 4,  0.2 # RK steps per interval vs time step
        X = np.asarray(cur_state) if isinstance(cur_state, list) else cur_state

        for j in range(M):
            if np.any(t_span): # integrate for this much time steps
                hh = (t_span[1]-t_span[0])/10/M
                for h in np.arange(t_span[0], t_span[1], hh):
                    k1 = self.dynamics(X)
                    k2 = self.dynamics(X + h/2 * k1)
                    k3 = self.dynamics(X + h/2 * k2)
                    k4 = self.dynamics(X + h * k3)
                    X  = X+(h/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                k1 = self.dynamics(X)
                k2 = self.dynamics(X + h/2 * k1)
                k3 = self.dynamics(X + h/2 * k2)
                k4 = self.dynamics(X + h * k3)

                X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

        return X

    def dissipation(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Parameters
            ==========
                dim: The dissipation of the Hamiltonian on
                the grid (see 5.11-5.12 of O&F).

                t, data, derivMin, derivMax, schemeData: other parameters
                here are merely decorators to  conform to the boilerplate
                we use in the LevelSetPy toolbox.
        """
        assert dim>=0 and dim <3, "Dubins vehicle dimension has to between 0 and 2 inclusive."

        if dim==0:
            return cp.abs(self.v_e - self.v_p * cp.cos(self.grid.xs[2])) + cp.abs(self.w(1) * self.grid.xs[1])
        elif dim==1:
            return cp.abs(self.v_p * cp.sin(self.grid.xs[2])) + cp.abs(self.w(1) * self.grid.xs[0])
        elif dim==2:
            return self.w_e + self.w_p
