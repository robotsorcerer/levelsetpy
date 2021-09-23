import numpy as np
from Utilities import *
from ValueFuncs import *
from Grids import processGrid
from BoundaryCondition import addGhostPeriodic
from scipy.interpolate import RegularGridInterpolator
import logging

logger = logging.getLogger(__name__)

def proj(g, data, dimsToRemove, xs=None, NOut=None, process=True):
    """
        [gOut, dataOut] = proj(g, data, dims, xs, NOut)
        Projects data corresponding to the grid g in g.dim dimensions, removing
        dimensions specified in dims. If a point is specified, a slice of the
        full-dimensional data at the point xs is taken.
        Inputs:
        g            - grid corresponding to input data
        data         - input data
        dimsToRemove - vector of len g.dim specifying dimensions to project
                    for example, if g.dim = 4, then dims = [0 0 1 1] would
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
    """
    # print(f'dimsToRemove: {dimsToRemove} {dimsToRemove.shape}')
    # Input checking
    if len(dimsToRemove) != g.dim:
        logger.fatal('Dimensions are inconsistent!')

    if np.count_nonzero(np.logical_not(dimsToRemove)) == g.dim:
        gOut = g
        dataOut = data
        logger.warning('Input and output dimensions are the same!')
        return gOut, dataOut

    # By default, do a projection
    if not xs:
        xs = 'min'

    # If a slice is requested, make sure the specified point has the correct
    # dimension
    if isnumeric(xs) and len(xs) != np.count_nonzero(dimsToRemove):
        logger.fatal('Dimension of xs and dims do not match!')

    if NOut is None:
        NOut = g.N[np.logical_not(dimsToRemove)]
        if NOut.ndim < 2:
            NOut = expand(NOut, 1)
    # print(f'NOut: {NOut} {NOut.shape}')
    dataDims = data.ndim
    if np.any(data) and np.logical_not(dataDims == g.dim or dataDims == g.dim+1) \
        and not isinstance(data, list):
        logger.fatal('Inconsistent input data dimensions!')

    # Project data
    if dataDims == g.dim:
        gOut, dataOut = projSingle(g, data, dimsToRemove, xs, NOut, process)
    else:
        # Project grid
        gOut, _ = projSingle(g, None, dimsToRemove, xs, NOut, process)

        # Project data
        if isinstance(data, list):
            numTimeSteps = len(data)
        else:
            numTimeSteps = data.shape[dataDims-1]

        dataOut = [] #np.zeros( NOut.T.shape + (numTimeSteps,) )

        for i in range(numTimeSteps):
            if isinstance(data, list):
                _, res = projSingle(g, data[i], dimsToRemove, xs, NOut, process)
                dataOut.append(res)
            else:
                _, res = projSingle(g, data, dimsToRemove, xs, NOut, process)
                dataOut.append(res)

            dataO.append(dataOut)

        dataOut = np.asarray(dataOut)

    return gOut, dataOut



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
                     for example, if g.dim = 4, then dims = [0 0 1 1] would
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

     Original by Sylvia;
     Python by Lekan July 29. 2021
    """

    # create ouptut grid by keeping dimensions that we are not collapsing
    if not g:
        if not isnumeric(xs) or xs!='max' and xs!='min':
            logger.fatal('Must perform min or max projection when not specifying grid!')
    else:
        dims = dims.astype(bool)
        gOut = Bundle(dict(
                dim = np.count_nonzero(np.logical_not(dims)),
                min = None,
                max = None,
                bdry = None,
        ))
        ming = g.min[np.logical_not(dims)]
        maxg = g.max[np.logical_not(dims)]
        bdrg = np.asarray(g.bdry)[np.logical_not(dims)]
        # print('bdrg: ', bdrg)
        gOut.min = ming if ming.ndim==2 else expand(ming, 1)
        gOut.max = maxg if maxg.ndim==2 else expand(maxg, 1)
        gOut.bdry = bdrg if bdrg.ndim==2 else expand(bdrg, 1)

        if numel(NOut) == 1:
            gOut.N = NOut*ones(gOut.dim, 1, order=ORDER_TYPE).astype(np.int64)
        else:
            gOut.N = NOut


        if process:
            gOut = processGrid(gOut)
        # print( 'g.vs aft proc', [x.shape for x in gOut.vs])

        # Only compute the grid if value function is not requested
        if not np.any(data) or data is None:
            return gOut, None

    # 'min' or 'max'
    if isinstance(xs, str):
        dimsToProj = np.nonzero(dims)[0]

        for i in range(len(dimsToProj)-1, -1, -1):
            # print('dara: ', data.shape, dimsToProj, xs)
            if xs=='min':
                data = np.amin(data, axis=dimsToProj[i])
            elif xs=='max':
                data = np.amax(data, axis=dimsToProj[i])
            else:
                logger.fatal('xs must be a vector, ''min'', or ''max''!')

        dataOut = data
        return gOut, dataOut

    # Take a slice
    g, data = augmentPeriodicData(g, data)

    eval_pt = cell(g.dim, 1)
    xsi = 0
    # print(f'xs: {xs}')
    for i in range(g.dim):
        if dims[i]:
            # If this dimension is periodic, wrap the input point to the correct period
            if isfield(g, 'bdry') and id(g.bdry[i])==id(addGhostPeriodic):
                period = max(g.vs[i]) - min(g.vs[i])
                while xs[xsi] > max(g.vs[i]):
                    xs[xsi] -= period
                while xs[xsi] < min(g.vs[i]):
                    xs[xsi] += period
            eval_pt[i] = np.asarray([xs[xsi]], order=ORDER_TYPE)
            xsi += 1
        else:
            eval_pt[i] = g.vs[i].squeeze()

    # https://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy
    data_coords = tuple([x.squeeze() for x in g.vs])
    interp_func = RegularGridInterpolator(data_coords, data)
    points = np.meshgrid(*eval_pt, indexing='ij')
    flat = np.array([m.flatten(order='f') for m in points], order=ORDER_TYPE)
    temp = interp_func(flat.T).reshape(*points[0].shape, order=ORDER_TYPE)

    dataOut = copy.copy(temp.squeeze())
    temp = np.asarray(g.vs, dtype=np.object)[np.logical_not(dims)]
    data_coords = tuple([x.squeeze() for x in temp])
    interp_func = RegularGridInterpolator(data_coords, dataOut)

    flat = np.array([m.flatten(order=ORDER_TYPE) for m in gOut.xs])
    dataOut = interp_func(flat.T).reshape(*gOut.xs[0].shape, order=ORDER_TYPE)

    return gOut, dataOut
