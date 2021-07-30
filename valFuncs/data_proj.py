import numpy as np
from utils import *
from valFuncs import *
from grids import processGrid
import logging

logger = logging.getLogger(__name__)

def projSingle(g, data, dims, xs, NOut, process):
    """
     [gOut, dataOut] = proj(g, data, dims, xs, NOut)
       Projects data corresponding to the grid g in g.dim dimensions, removing
       dimensions specified in dims. If a point is specified, a slice of the
       full-dimensional data at the point xs is taken.

     Inputs:
       g       - grid corresponding to input data
       data    - input data
       dims    - vector of len g.dim specifying dimensions to project
                     For example, if g.dim = 4, then dims = [0 0 1 1] would
                     project the last two dimensions
       xs      - Type of projection (defaults to 'min')
           'min':    takes the union across the projected dimensions
           'max':    takes the intersection across the projected dimensions
           a vector: takes a slice of the data at the point xs
       NOut    - number of grid points in output grid (defaults to the same
                 number of grid points of the original grid in the unprojected
                 dimensions)
       process            - specifies whether to call processGrid to generate
                            grid points

     Outputs:
       gOut    - grid corresponding to projected data
       dataOut - projected data

     See proj_test.py

     Original by Sylvia; Python by Lekan July 29. 2021
    """

    # Create ouptut grid by keeping dimensions that we are not collapsing
    if not g:
        if not isinstance(xs, str) or xs!='max' and xs!='min'):
            logger.fatal('Must perform min or max projection when not specifying grid!')
    else:
        dims = dims.astype(bool)
        gOut.dim = np.count_nonzero(np.logical_not(dims));
        gOut.min = g.min[np.logical_not(dims)];
        gOut.max = g.max[np.logical_not(dims)];
        gOut.bdry = g.bdry[np.logical_not(dims)];

        if np.size(NOut) == 1:
            gOut.N = NOut*ones(gOut.dim, 1);
        else:
            gOut.N = NOut;

        # Process the grid to populate the remaining fields if necessary
        if process:
            gOut = processGrid(gOut);

        # Only compute the grid if value function is not requested
        if not data: #nargout < 2
            return

    # 'min' or 'max'
    if isinstance(xs, str):
        dimsToProj = np.nonzero(dims)[0];

        for i in range(len(dimsToProj)-1, -1, -1):
            if xs=='min':
                data = np.squeeze(np.min(data, dimsToProj[i]));
            elif xs=='max':
                data = np.squeeze(np.max(data, dimsToProj[i]));
            else:
                logger.fatal('xs must be a vector, ''min'', or ''max''!')

        dataOut = data;
        return

    # Take a slice
    # Preprocess periodic dimensions
    g, data = augmentPeriodicData(g, data);

    # temp = interpn(g.vs{1}, g.vs{2}, g.vs{3}, g.vs{4}, data, g.vs{1}, xs(1), ...
    #   g.vs{3}, xs(2));
    eval_pt = cell(g.dim, 1);
    xsi = 1;
    for i in range(g.dim):
        if dims[i]:
            # If this dimension is periodic, wrap the input point to the correct period
            if (g.bdry and isinstance(g.bdry[i], addGhostPeriodic)):
                period = max(g.vs[i]) - min(g.vs[i]);
                while xs[xsi] > max(g.vs[i]):
                    xs[xsi] -= period;
                while xs[xsi] < min(g.vs[i]):
                    xs[xsi] += period;
            eval_pt[i] = xs[xsi];
            xsi += 1;
        else:
            eval_pt[i] = g.vs[i];

    temp = np.interp(g.vs[:], data, eval_pt[:]);

    dataOut = temp.squeeze();

    # interpn(g.vs{1}, g.vs{3}, dataOut, gOut.xs{1}, gOut.xs{2})
    dataOut = np.interp(g.vs[np.logical_not(dims)], dataOut, gOut.xs[:]);
    return gOut, dataOut


def proj(g, data, dimsToRemove, xs=None, NOut=None, process=None):
    """
        [gOut, dataOut] = proj(g, data, dims, xs, NOut)
        Projects data corresponding to the grid g in g.dim dimensions, removing
        dimensions specified in dims. If a point is specified, a slice of the
        full-dimensional data at the point xs is taken.
        Inputs:
        g            - grid corresponding to input data
        data         - input data
        dimsToRemove - vector of len g.dim specifying dimensions to project
                    For example, if g.dim = 4, then dims = [0 0 1 1] would
                    project the last two dimensions
                    xs           - Type of projection (defaults to 'min')
                    'min':    takes the union across the projected dimensions
                    'max':    takes the intersection across the projected dimensions
                    a vector: takes a slice of the data at the point xs
                    NOut    - number of grid points in output grid (defaults to the same
                    number of grid points of the original grid in the unprojected
                    dimensions)
                    process            - specifies whether to call processGrid to generate
                            grid points
                            Outputs:
                            gOut    - grid corresponding to projected data
                            dataOut - projected data
                            See proj_test.m
    """
    # Input checking
    if len(dimsToRemove) != g.dim:
        logger.fatal('Dimensions are inconsistent!')

    if np.count_nonzero(np.logical_not(dimsToRemove)) == g.dim:
        gOut = g;
        dataOut = data;
        logger.warning('Input and output dimensions are the same!')
        return

    # By default, do a projection
    if not xs:
        xs = 'min';

    # If a slice is requested, make sure the specified point has the correct
    # dimension
    if np.char.isnumeric(xs) and len(xs) != np.count_nonzero(dimsToRemove):
        logger.fatal('Dimension of xs and dims do not match!')

    if not NOut:
        NOut = g.N[np.logical_not(dimsToRemove)]

    if not process:
        process = True;

    dataDims = numDims(data);
    if np.any(data) and np.logical_not(dataDims == g.dim or dataDims == g.dim+1) \
        and not isinstance(data, list):
        logger.fatal('Inconsistent input data dimensions!')

    # Project data
    if dataDims == g.dim:
        gOut, dataOut = projSingle(g, data, dimsToRemove, xs, NOut, process);

    else: # dataDims == g.dim + 1
        # Project grid
        gOut = projSingle(g, [], dimsToRemove, xs, NOut, process);

        # Project data
        if isinstance(data, list):
            numTimeSteps = len(data)
        else:
            numTimeSteps = data.shape[dataDims]
            colonsIn = [[':'] for i in range(g.dim)] #repmat({':'}, 1, g.dim);

        dataOut = zeros([NOut.T, numTimeSteps]);
        colonsOut =  [[':'] for i in range(g.dim)] #repmat({':'}, 1, gOut.dim);

        for i in range(numTimeSteps):
            if isinstance(data, list):
                _, dataOut[colonsOut[:],i] = projSingle(g, data[i], dimsToRemove, xs, NOut, process);
            else:
                _, dataOut[colonsOut[:],i] = projSingle(g, data[colonsIn[:],i], dimsToRemove, xs, NOut, process);

    return gOut, dataOut
