__all__ = ["RocketSystemRel"]


__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

__date__        = "Dec. 21, 2021"
__comment__     = "Two Rockets in Relative Coordinates"

import numpy as np

from LevelSetPy.Utilities.matlab_utils import isColumnLength, isvector

class RocketSystemRel():
    def __init__(self, grid, u_bound=5, w_bound=5, x=None, a=32, g=64):
        """
            Rockets in relative coordinates.
            Please consult Dreyfus, 1964 for a detailed reference.

            The equations of motion are adopted from Dreyfus' construction. In relative 
            coordinates between the two rockets, we have:
            
                    &\dot{x} = a(cos(u_p)+ cos(u_e)),\\
                    &\dot{z} = a(sin(u_p)+sin(u_e))-2g,\\
                    &\dot{\theta} = u_p - u_e

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
        self.u = lambda u: u*u_bound
        self.w = lambda w: w*w_bound
        self.a = a
        self.ap, self.ae = a, a
        self.g = g

        # set linear speeds
        self.u_e = self.u(u_bound)
        self.u_p = self.u(-u_bound)

    def hamiltonian(self, t, value, value_derivs, finite_diff_bundle):
        """
            H = -a p_1 \cos θ - p_2(g - a -asin θ) - \bar{u} | p_1 x + p_3 | + 
                                underline{u} | p_2 x + p_3|

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
        p1, p2, p3 = value_derivs[0], value_derivs[1], value_derivs[2]

        p1_coeff = -self.a*np.cos(self.grid.xs[2]) 
        p2_coeff = self.g - self.a - self.a*np.sin(self.grid.xs[2])
        p31_coeff = np.abs(p1*self.grid.xs[0] + p3)
        p32_coeff = np.abs(p2*self.grid.xs[0] + p3)

        Hxp = p1*p1_coeff + p2*p2_coeff - self.u_p*p31_coeff + self.u_p*p32_coeff

        return Hxp

    def dissipation(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Parameters
            ==========
                dim: The dissipation of the Hamiltonian on
                the grid (see 5.11-5.12 of O&F).

                t, data, derivMin, derivMax, schemeData: other parameters
                here are merely decorators to  conform to the boilerplate
                we use in the levelsetpy toolbox.
        """
        assert dim>=0 and dim <3, "Dubins vehicle dimension has to between 0 and 2 inclusive."

        if dim==0:
            return np.abs(-self.a*np.cos(self.grid.xs[2])) + np.abs(self.u_e*self.grid.xs[0])
        elif dim==1:
            return np.abs(self.g - self.a -self.a*np.sin(self.grid.xs[2])) + np.abs(self.u_p*self.grid.xs[0]) # - self.a - self.g)
        elif dim==2:
            return np.abs(self.u_p + self.u_e)

    def dynamics(self):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x} = a_p cos θ + u_e x
            \dot{z} = a_p sin θ + a_e + u_e x - g
            \dot{θ} = u_p - u_e
        """

        x1 = self.grid.xs[0]
        x2 = self.grid.xs[1]
        x3 = self.grid.xs[2]

        xdot = [
                -self.ae + self.ap * np.cos(x3) + self.u_e * x2,
                -self.ap * np.sin(x3) - self.u_e * x1,
                -self.u_p - self.u_e 
        ]

        return xdot
