from Utilities import *
from SpatialDerivative import upwindFirstWENO5

def computeGradients(g, data, dims=None, derivFunc=None):
    """
     [derivC, derivL, derivR] = computeGradients(g, data, derivFunc, upWind)

     Estimates the costate p at position x for cost function data on grid g by
     numerically taking partial derivatives along each grid direction.
     Numerical derivatives are taken using the levelset toolbox

     Inputs: grid      - grid structure
             data      - array of g.dim dimensions containing function values
             derivFunc - derivative approximation function (from level set
                         toolbox)
     Output: derivC    - (central) gradient in a g.dim by 1 vector
             derivL    - left gradient
             derivR    - right gradient

    Lekan Molu, August 11, 2021
    """
    if not dims:
        dims = zeros(g.dim, 1)
        dims.fill(True)

    if not derivFunc:
        # this computes the gradients of the value function
        derivFunc = upwindFirstWENO5


    # Go through each dimension and compute the gradient in each dim
    derivC = cell(g.dim, 1)

    if numDims(data) == g.dim:
        tau_length = 1
    elif numDims(data) == g.dim + 1:
        tau_length = size(data)
        tau_length = tau_length[-1]
    else:
        error('Dimensions of input data and grid don''t match!')

    # Just in case there are NaN values in the data (usually from TTR functions)
    numInfty = 1e6
    data[np.isnan(data)] = numInfty

    # Just in case there are inf values
    data[np.isinf(data)] = numInfty

    for i in range(g.dim):
        if dims[i]:
            derivC[i] = zeros(size(data))

            ## data at a single time stamp
            if tau_length == 1:
                # Compute gradient using upwinding weighted essentially non-oscillatory differences
                derivL, derivR = derivFunc(g, data, i)

                # Central gradient
                derivC[i] = 0.5*(derivL + derivR)
            else:
                ## data at multiple time stamps
                for t in range(tau_length):
                    derivL, derivR = derivFunc(g, data[t, ...], i)
                    derivC[i][t,...] = 0.5*(derivL + derivR)

            # Change indices where data was nan to nan
            derivC[i][nanInds] = np.nan

            # Change indices where data was inf to inf
            derivC[i][infInds] = np.inf

    return derivC, derivL, derivR
