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

import cupy as cp
import numpy as np

from LevelSetPy.Utilities.matlab_utils import isColumnLength, isvector

class RocketSystemRel():
    def __init__(self, grid, u_bound=5, w_bound=5, x=None, a=32, g=64):
        """
            Dubins Vehicle Dynamics in relative coordinates.
            Please consult Merz, 1972 for a detailed reference.

            The equations of motion are adopted from Dreyfus' construction as follows:

                &\dot{y}_1 = y_3,          &\dot{y}_5 = y_7, \\
                &\dot{y}_2 = y_4,          &\dot{y}_6 = y_8, \\
                &\dot{y}_3 = a\cos(u),     &\dot{y}_7 = a\cos(v), \\
                &\dot{y}_4 = a\sin(u) - g, &\dot{y}_8 = a\sin(v) - g.

            where $u(t), t \in [-T,0]$ is the controller under the coercion of the evader and
             $v(t), t \in [-T,0]$ is the controller under the coercion of the pursuer i.e.
             the pursuer is minimizing while the evader is maximizing. The full state dynamics
             is given by

            \dot{y} = \left(\begin{array}{c}
                            \dot{y}_1 & \dot{y}_2 & \dot{y}_3 & \dot{y}_4 \\
                            \dot{y}_5 & \dot{y}_6 & \dot{y}_7 &\dot{y}_8 \\
                            \end{array}
                        \right)^T =
                            \left(\begin{array}{c}
                            y_3 & y_4 & a\cos(u) & a\sin(u) - g &
                            y_7 & y_8 & a\cos(v) & a\sin(v) - g
                            \end{array}\right).

            In relative coordinates between the two rockets, we have
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
            assert isinstance(x, np.ndarray) or isinstance(x, cp.ndarray), "initial state must either be a numpy or cupy array."
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
        self.u_p = self.u(u_bound)

    def hamiltonian(self, t, value, value_derivs, finite_diff_bundle):
        """
            H = p_1 [u_e - u_p cos(x_3)] - p_2 [u_p sin x_3] \
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
        p1, p2, p3 = value_derivs[0], value_derivs[1], value_derivs[2]
        p1_coeff = self.a - self.a * cp.cos(self.grid.xs[2])
        p2_coeff = self.a * cp.sin(self.grid.xs[2])

        Hxp = -self.a*p1*cp.cos(self.grid.xs[2]) + p2*(self.g-self.a-self.a*cp.sin(self.grid.xs[2])) \
              -self.u_e*cp.abs(p1*self.grid.xs[0]+p2*self.grid.xs[0]+p3) \
                - self.u_p*cp.abs(p3)

        # Hxp = p1 * p1_coeff - p2 * p2_coeff - self.w(1)*cp.abs(p1*self.grid.xs[1] - \
        #         p2*self.grid.xs[0] - p3) + self.u(1) * cp.abs(p3)

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
            return cp.abs(self.a*cp.cos(self.grid.xs[2]))+cp.abs(+self.u_e*self.grid.xs[0])
        elif dim==1:
            return cp.abs(self.g)+cp.abs(self.a)+cp.abs(self.a*cp.sin(self.grid.xs[2]))+cp.abs(self.u_e*self.grid.xs[0]) #
        elif dim==2:
            return cp.abs(self.u_p) + cp.abs(self.u_e)

    def dynamics(self):
        """
            Computes the Dubins vehicular dynamics in relative
            coordinates (deterministic dynamics).

            \dot{x}_1 = -u_e + u_p cos x_3 + w_e x_2
            \dot{x}_2 = -u_p sin x_3 - w_e x_1
            \dot{x}_3 = -w_p - w_e
        """

        # xdot = [
        #             self.a*cp.cos(self.grid.xs[2])+self.u_e*self.grid.xs[0],
        #             self.a*cp.sin(self.grid.xs[2])+self.a+self.u_e*self.grid.xs[0]-self.g,
        #             self.u_p - self.u_e,
        #         ]
        x1 = self.grid.xs[0]
        x2 = self.grid.xs[1]
        x3 = self.grid.xs[2]

        xdot = [
                -self.ae + self.ap * np.cos(x3) + self.u_e * x2,
                -self.ap * np.sin(x3) - self.u_e * x1,
                -self.u_p - self.u_e # pursuer minimizes
        ]

        return xdot
