from Utilities import *
from ValueFuncs import *

def eval_u(gs, datas, xs, interp_method='linear'):
    """
     v = eval_u(g, datas, x)
       Computes the interpolated value of the value function data at the
       states xs

     Inputs:
       Option 1: Single grid, single value function, multiple states
         gs    - a single grid structure
         datas - a single matrix (look-up table) representing the value
                 function
         xs    - set of states each row is a state

       Option 2: Single grid, multiple value functions, single state
         gs    - a single grid structure
         datas - a list of matrices representing the value function
         xs    - a single state

       Option 3: Multiple grids, value functions, and states. The number of
                 grids, value functions, and states must be equal under this
                 option
         gs    - list containing multiple grid structures
         datas - list containing multiple matrices of value functions
         xs    - list containing multiple states
    Output:
        Returns the value function, v

     Lekan Molux, 2021-08-09
     """

    if isbundle(gs) and isinstance(datas, np.array) and len(xs.shape)>=2:
        # Option 1
        v = eval_u_single(gs, datas, xs, interp_method)
    elif isbundle(gs) and iscell(datas) and isvector(xs):
        # Option 2
        v = cell(len(datas), 1)
        for i in range(len(datas)):
            v[i] = eval_u_single(gs, datas[i], xs, interp_method)
        v = np.array(v)
    elif iscell(gs) and iscell(datas) and iscell(xs):
        # Option 3
        v = cell(len(gs), 1)
        for i in range(len(gs)):
            v[i] = eval_u_single(gs[i], datas[i], xs[i], interp_method)
        v = np.array(v)
    else:
        error('Unrecognized combination of input data types!')

    return v

def  eval_u_single(g, data, x, interp_method):
    """
     v = eval_u_single(g, data, x)
       Computes the interpolated value of a value function data at state x

     Inputs:
       g       - grid
       data    - implicit function describing the set
       x       - points to check each row is a point

     OUTPUT
       v:  value at points x

     Python Ver: August 9, 2021

    """
    # If the number of columns does not match the grid dimensions, try taking
    # transpose
    if size(x, 2) != g.dim:
      x = x.T

    g, data = augmentPeriodicData(g, data)

    ## Dealing with periodicity
    for i in range(g.dim):
        if (isfield(g, 'bdry') and g.bdry[i]==addGhostPeriodic):
            # Map input points to within grid bounds
            period = max(g.vs[i]) - min(g.vs[i])

            i_above_bounds = x[:,i] > max(g.vs[i])
            while np.any(i_above_bounds):
                x[i_above_bounds, i] -= period
                i_above_bounds = x[:,i] > max(g.vs[i])

            i_below_bounds = x[:,i] < min(g.vs[i])
            while np.any(i_below_bounds):
                x[i_below_bounds, i] += period
                i_below_bounds = x[:,i] < min(g.vs[i])

    ## Interpolate
    # Input checking
    x = checkInterpInput(g, x)

    # eg. v = interpn(g.vs{1}, g.vs{2}, data, x(:,1), x(:,2), interp_method)
    interpn_argin_x = []

    for i in range(g.dim):
        interpn_argin_x.append(x[:,i])

    v = np.interp(g.vs[:], data, interpn_argin_x[:])

    return v
