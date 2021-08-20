import copy
import numpy as np
from utils import *
from DynamicalSystems import DynSys

class DubinsCar(DynSys):
    def __init__(self, **kwargs):
        # super().__init__(kwargs)
        self.__doc__ = """
               obj = DubinsCar(x, wMax, speed, dMax, dims)
                   Dubins Car class

                kwargs = {
                            wRange=[-wMax, wMax], speed=5, \
                            dRange=zeros(3,2), dims=np.arange(3), \
                            end=None, nu = 1, nd = 3, x = None, nx=3, \
                            u = None, xhist=None, uhist=None, pdim=None, \
                            vdim, hdim, hpxpy=None, hpxpyhist=None, \
                            hvxvy=None, hvxvyhist, hpv = cell(2,1), \
                            hpvhist = cell(2,1), data = None
                    }

               Dynamics:
                  \dot{x}_1 = v * cos(x_3) + d1
                  \dot{x}_2 = v * sin(x_3) + d2
                  \dot{x}_3 = u
                       u \in [-wMax, wMax]
                       d \in [-dMax, dMax]

               Inputs:
                 x      - state: [xpos; ypos]
                 thetaMin   - minimum angle
                 thetaMax   - maximum angle
                 v - speed
                 dMax   - disturbance bounds

               Output:
                 obj       - a DubinsCar2D object
            """
        # self.dyn_sys = DynSys(kwargs)
        DynSys.__init__(self, kwargs)

    def update_dynamics(self, x, u, d=zeros(3, 1), **kwargs):
        """
        ^^^This is the ode function for the Dynamics of
            the Dubins Car. This function takes the base
            parameters of the dubins car model <computed
            in our `__init__` function> and returns the
            o.d.e of the LHS according to:

                \dot{x}_1 = v * cos(x_3) + d1
                \dot{x}_2 = v * sin(x_3) + d2
                \dot{x}_3 = w

            There is an implicit assumption that the user is working
             in relative coordinates between the cars.

            Inputs:
                x: 3-D state, or N X 3D state for N evolved states
                u: control==w
                d: disturbance

            Function parameters:
                v: linear speed
                w: orientation of the cars w.r.t one another

            This is optional:
                kwargs: dictionary containing
                [wRange, speed, dRange, dims]

                kwargs must be suppied if any of these parameters
                must change during simulation.

             Adopted from Mo Chen, 2016-06-08
             Lekan Molu, 2021-08-08
        """
        x = to_column_mat(x)

        if T==0:
            x1 = x0;
            return x1

        # Basic vehicle properties
        self.dims = kwargs['dims'] if 'dims' in kwargs else self.dims
        self.dims = kwargs['dims'] if 'dims' in kwargs else self.dims
        # assert self.dims.shape[0]==1, "first dimension of dims array can not be non-singleton"
        self.pdim = np.hstack([np.nonzero(self.dims == 1), np.nonzero(self.dims == 2)]) # Position dimensions
        #obj.hdim = find(dims == 3);   # Heading dimensions
        self.nx = len(self.dims);

        if numel(x) != self.nx:
            error(f'Initial state dim, x_0: {numel(x)} != obj.nx: {self.nx}!');

        self.nu = 1;
        self.nd = 3;

        self.x = x;
        self.xhist = self.x;

        x0 = kwargs['x0'] if 'x0' in kwargs else None
        if not x0:
            x0 = self.x

        if not np.any(u):
            logger.warn(f'Controls u is empty')
            return self.x

        if np.any(np.isnan(u)):
            logger.warn(f'Controls u contain NaNs')
            return self.x

        u = to_column_mat(u)

        if 'wRange' in kwargs: # if we want to change the default
            self.wRange = kwargs['wRange'] # angke bounds
        #self.thetaMax = thetaMax;
        if 'speed' in kwargs:
            self.speed = kwargs['speed'] # speed, v
        if 'dRange' in kwargs:
            self.dRange = kwargs['dRange'] # disturbance bounds
        if 'dims' in kwargs:
            self.dims = kwargs['dims'] # dims should be 3 in length
        # integrate=kwargs['integrate'] if 'integrate' in kwargs else None

        def dynamics_helper(x, u, d, dims, dim):
            if dim==1:
                dx = self.speed * np.cos(x[dims==2]) + d[0]
            elif dim==2:
                dx = self.speed * np.sin(x[dims==2]) + d[1]
            elif dim==3:
                dx = u + d[2]
            else:
                error('Only dimension 1-3 are defined for dynamics of DubinsCar!')
            return dx

        if iscell(self.x):
            # this for aggregated evolved states
            dx = cell(len(self.dims), 1);

            for i in range(len(self.dims)):
                dx[i] = dynamics_helper(x, u, d, self.dims, self.dims[i]);
        else:
            # this for a single state
            dx = zeros(self.nx, 1);

            dx[0] = self.speed * np.cos(x[2]) + d[0];
            dx[1] = self.speed * np.sin(x[2]) + d[1];
            dx[2] = u + d[2]

        return np.array(dx)

    def get_opt_u(self, deriv, uMode='min', t=None, y=None):
        "Returns optimal control law"

        if not iscell(deriv):
            deriv = cell(deriv, 1)

        ## Optimal control: check these formulas
        if uMode=='max':
            uOpt = (deriv[self.dims==3]>=0)@self.wRange[1] + (deriv[self.dims==3]<0)@(self.wRange[0]);
        elif uMode=='min':
            uOpt = (deriv[self.dims==3]>=0)@(self.wRange[0]) + (deriv[self.dims==3]<0)@self.wRange[1];
        else:
            error('Unknown control Mode!')

        return uOpt

    def get_opt_v(self, deriv, dMode='max', t=None, y=None):
        """Returns optimal disturbance
        """
        if not iscell(deriv):
            deriv = cell(deriv, 1)

        vOpt = zeros(3, 1);

        ## Optimal control
        if dMode=='max':
          for i in range(3):
              if np.any(self.dims[self.dims == i]):
                  vOpt[i] = (deriv[self.dims==i]>=0)@self.dRange[1][i] + \
                            (deriv[self.dims==i]<0)@(self.dRange[0][i])

        elif dMode=='min':
          for i in range(3):
              if np.any(self.dims[self.dims == i]):
                  vOpt[i] = (deriv[self.dims==i]>=0)@self.dRange[0][i] + \
                            (deriv[self.dims==i]<0)@(self.dRange[1][i])
        else:
            error('Unknown dMode!')

        return vOpt
