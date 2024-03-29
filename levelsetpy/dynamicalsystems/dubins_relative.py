__all__ = ["DubinsVehicleRel"]


__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

__date__        = "Dec. 21, 2021"
__comment__     = "Two Dubins Vehicle in Relative Coordinates"

import cupy as cp
import numpy as np

from levelsetpy.utilities.matlab_utils import isColumnLength, isvector

class DubinsVehicleRel():
    def __init__(self, grid, u_bound=5, w_bound=5, x=None):
        """
            Dubins Vehicle Dynamics in relative coordinates.
            Please consult Merz, 1972 for a detailed reference.

            Dynamics:

                \dot{x}_1 = -v_e + v_p cos x_3 + w_e x_2
                \dot{x}_2 = -v_p sin x_3 - w_e x_1
                \dot{x}_3 = -w_p - w_e

            Parameters
            ==========
                grid: an np.meshgrid state space on which we are
                resolving this vehicular dynamics.
                u_bound: absolute value of the linear speed of the vehicle.
                w_bound: absolute value of the angular speed of the vehicle.
        """

        self.grid        = grid

        if x:
            assert isinstance(x, np.ndarray) or isinstance(x, np.ndarray), "initial state must either be a numpy or cupy array."
            r, c = x.shape
            if r<c:
                # turn to column vector
                x = x.T

            self.cur_state   = x
        else:
            self.cur_state = self.grid.xs
        self.v = lambda u: u*u_bound
        self.w = lambda w: w*w_bound

        # set linear speeds
        if not np.isscalar(u_bound) and len(u_bound) > 1:
            self.v_e = self.v(u_bound)
            self.v_p = self.v(-u_bound)
        else:
            self.v_p = self.v(u_bound)
            self.v_e = self.v(u_bound)

        # set angular speeds
        if not np.isscalar(w_bound) and len(w_bound) > 1:
            self.w_e = self.w(u_bound)
            self.w_p = self.w(-u_bound)
        else:
            self.w_p = self.w(u_bound)
            self.w_e = self.w(u_bound)

    def hamiltonian(self, t, data, value_derivs, finite_diff_bundle):
        """
            H = p_1 [v_e - v_p cos(x_3)] - p_2 [v_p sin x_3] \
                   - w | p_1 x_2 - p_2 x_1 - p_3| + w |p_3|

            Parameters
            ==========
            value: Value function at this time step, t
            value_derivs: Spatial derivatives (finite difference) of
                        value function's grid points computed with
                        upwinding.
            finite_diff_bundle: Bundle for finite difference function
                .innerData: Bundle with the following fields:
                    .partialFunc: RHS of the o.d.e of the system under consideration
                        (see function dynamics below for its impl).
                    .hamFunc: Hamiltonian (this function).
                    .dissFunc: artificial dissipation function.
                    .derivFunc: Upwinding scheme (upwindFirstENO2).
                    .innerFunc: terminal Lax Friedrichs integration scheme.
        """
        p1, p2, p3  = value_derivs[0], value_derivs[1], value_derivs[2]
        p1_coeff    = self.v_e - self.v_p * np.cos(self.grid.xs[2])
        p2_coeff    = self.v_p * np.sin(self.grid.xs[2])

        Hxp = p1 * p1_coeff - p2 * p2_coeff - self.w(1)*np.abs(p1*self.grid.xs[1] - \
                p2*self.grid.xs[0] - p3) + self.w(1) * np.abs(p3)

        return Hxp

    def dissipation(self, t, data, derivMin, derivMax, schemeData, dim):
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
            return np.abs(self.v_e - self.v_p * np.cos(self.grid.xs[2])) + np.abs(self.w(1) * self.grid.xs[1])
        elif dim==1:
            return np.abs(self.v_p * np.sin(self.grid.xs[2])) + np.abs(self.w(1) * self.grid.xs[0])
        elif dim==2:
            return self.w_e + self.w_p

    def dynamics(self):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x}_1 = -v_e + v_p cos x_3 + w_e x_2
            \dot{x}_2 = -v_p sin x_3 - w_e x_1
            \dot{x}_3 = -w_p - w_e
        """
        x1 = self.grid.xs[0]
        x2 = self.grid.xs[1]
        x3 = self.grid.xs[2]

        xdot = [
                -self.ve + self.vp * np.cos(x3) + self.we * x2,
                -self.vp * np.sin(x3) - self.we * x1,
                -self.wp - self.we # pursuer minimizes
        ]

        return xdot

