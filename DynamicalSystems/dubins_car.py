from .dyn_sys import DynSys
import copy
import numpy as np
from Utilities import *
from ExplicitIntegration.runge_kutta4 import dynamics_RK4

class DubinsCar(DynSys):
    def __init__(self, x, wRange, speed, dRange=None, dims=None):
        """
            Wrapper around Sylvia's Dubins Car Class
            dRange must be a {2X3} vector
            or a {1X3} vector
        """
        # self.x = to_column_mat(x)
        self.x = x
        # Angle bounds
        if wRange is None:
            self.wRange = np.array([[-1, 1]]).T
        elif numel(wRange) == 2:
            self.wRange = wRange
        else:
            self.wRange = [-wRange, wRange] #np.array([[-wRange, wRange]]).T

        self.speed = 5 if not speed else speed # Constant speed = speed

        # Disturbance
        if dRange is None:
            self.dRange = np.zeros((2, 3))
        elif np.any(dRange.shape)==1:
            # make it 2 x 3 by stacking
            self.dRange = np.vstack([-dRange, dRange])
        else:
            self.dRange = dRange

        # Dimensions that are active
        self.dims = np.arange(3) if dims is None else dims

        pdim = np.where((self.dims==0) | (self.dims==1))
        DynSys.__init__(self, nx=len(self.dims), nu=1,nd=3, uhist=[],
                                x = self.x, xhist=self.x, pdim=pdim
                               )

    def dynamics(self, t, x, u, d=None):
        # Dynamics of the Dubins Car
        #    \dot{x}_1 = v * cos(x_3) + d1
        #    \dot{x}_2 = v * sin(x_3) + d2
        #    \dot{x}_3 = w
        #   Control: u = w;
        # print('x in dyna: ', [a.shape for a in x])
        # import time
        # time.sleep(5)
        # debug(f'x: {x}')
        if not d:
            d = np.zeros((3,1))
        if iscell(x):
            dx = cell(len(self.dims), 1)
            for i in range(len(self.dims)):
                dx[i] = self.dynamics_helper(x, u, d, self.dims, self.dims[i])
        else:
            dx = np.zeros((self.nx, 1), dtype=np.float64)
            # print(f'dx in dyna: {dx.shape}, x: {x.shape}, d: {d.shape}')
            dx[0] = self.speed * np.cos(x[2]) + d[0]
            dx[1] = self.speed * np.sin(x[2]) + d[1]
            dx[2] = u + d[2]
            # info(f'dx in dynamics return: {[x.shape for x in dx]}')
        return dx

    def dynamics_helper(self, x, u, d, dims, dim):
        d = np.asarray(d)
        # print('x in dyna_help: ', [a.shape for a in x])
        # print(f'd in dyna_help: {d.shape}')
        # print(f'speed in dyna_help: {self.speed}')
        # print('u in dyna_help: ', u)

        if dim==0:
            dx = self.speed * np.cos(x[2]) + d[0]
        elif dim==1:
            dx = self.speed * np.sin(x[2]) + d[1]
        elif dim==2:
            dx = u + d[2]
        else:
            error('Only dimension 1-3 are defined for dynamics of DubinsCar!')
        # if isinstance(dx, np.ndarray):
        #     print(f'dx in dyna_help: {dx.shape}')
        # else:
        #     print(f'dx in dyna_help: {dx}')
        # print()
        # import time
        # time.sleep(1)
        return dx

    def get_opt_u(self, t=0, deriv=0, uMode='min', y=0):
        "Returns optimal control law; t, y=None by default"

        if numel(deriv)==1:
            deriv = cell(deriv, 1)
        ## Optimal control
        if uMode=='max':
            uOpt = (deriv[2]>=0)*self.wRange[1] + (deriv[2]<0)*(self.wRange[0]);
        elif uMode=='min':
            uOpt = (deriv[2]>=0)*(self.wRange[0]) + (deriv[2]<0)*self.wRange[1]
            # print(uOpt.shape, 'uOpt')
        else:
            error('Unknown control Mode!')

        return uOpt

    def get_opt_v(self, t=0, deriv=0, dMode='max', y=0):
        """Returns optimal disturbance
        """
        if numel(deriv)==1:
            deriv = cell(deriv, 1)

        vOpt = []

        ## Optimal control
        if dMode=='max':
          for i in range(3):
              if np.any(self.dims == i):
                  vOpt.append((deriv[i]>=0)*self.dRange[1][i] + \
                            (deriv[i]<0)*(self.dRange[0][i]))

        elif dMode=='min':
          for i in range(3):
              if np.any(self.dims[self.dims == i]):
                  vOpt.append((deriv[i]>=0)*self.dRange[0][i] + \
                            (deriv[i]<0)*(self.dRange[1][i]))
        else:
            warn(f'Unknown dMode: {dMode}!')

        return vOpt

    def update_state(self, u, T=0, x0=None, d=[]):
        # Updates state based on control
        #
        # Inputs:   obj - current quardotor object
        #           u   - control (defaults to previous control)
        #           T   - duration to hold control
        #           x0  - initial state (defaults to current state if set to [])
        #           d   - disturbance (defaults to [])
        #
        # Outputs:  x1  - final state
        if x0 is None:
            x0 = self.x

        if T == 0:
            return x0

        if u is None:
            return x0

        if np.isnan(u):
            warn(f'u is Nan')
            return x0

        # print(f'u: {u}, {u.shape}')
        if numel(u) > 1 and isinstance(u, np.ndarray) and u.shape[0]<u.shape[1]:
            u = u.T

        x0 = x0.T if x0.shape[0]<x0.shape[1] else x0
        # print(f'x b4 RK4: {x0.shape}')
        if not np.any(d):
            x = self.integrate_dynamics([0, T], x0, u, [])
        else:
            x = self.integrate_dynamics([0, T], x0, u, d)
        x1 = np.asarray(x)
        # print(f'x aft RK4: {x1.shape}')

        self.x = x1
        self.u = u

        # print(f'self.xhist: {self.xhist.shape}')
        # print(f'self.uhist: {self.uhist.shape}, u: {u}')
        self.xhist = np.hstack((x1, self.xhist))
        self.uhist.append(u)

        return x1

    def integrate_dynamics(self, tspan, x, u, v):
        """
        # RK4 integrator for a time-invariant dynamical system under a control, u,
        and disturbance, v.

        # See https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

        This impl adopted from unstable-zeros's learning CBFs example for two airplanes

        https://github.com/unstable-zeros/learning-cbfs/blob/master/airplane_example/learning_cbfs_airplane.ipynb

        This function must be called within a loop for a total of N
        steps of integration, Obviously, the smallet the value of T, the better

        Inputs:
            self.dynamics: Right Hand Side of Ode function to be integrated
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

        # print(f'X: {X.shape}, h: {h}')
        for j in range(M):
            if np.any(tspan): # integrate for this much time steps
                for h in np.arange(tspan[0], tspan[1], hh):
                    k1 = self.dynamics(None, X, U, V)
                    # print(f'X + h/2: {(X + h/2).shape}, k1: {k1.shape}')
                    k2 = self.dynamics(None, X + h/2 * k1, U, V)
                    k3 = self.dynamics(None, X + h/2 * k2, U, V)
                    k4 = self.dynamics(None, X + h * k3, U, V)

                    X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)
            else:
                k1 = self.dynamics(None, X, U, V)
                k2 = self.dynamics(None, X + h/2 * k1, U, V)
                k3 = self.dynamics(None, X + h/2 * k2, U, V)
                k4 = self.dynamics(None, X + h * k3, U, V)

                X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

        return list(X)
