from Utilities import *

def  getOGPBounds(gBase, gMinIn, gMaxIn, padding):
    """"
        [gMinOut, gMaxOut] = getOGPBounds(gBase, gMinIn, gMaxIn)
        Returns grid bounds based on gBase, gMinIn, and gMaxIn such that if a new
        grid is constructed from gMinOut and gMaxOut, the grid points within the
        bounds of gBase would be the same.

        padding must be same dim as gMinIn, gMaxIn

        This is done without needing the actual grid points of gBase
    """

    # Compute or read grid spacing
    if isfield(gBase, 'dx'):
        dx = gBase.dx
    else:
        dx = np.divide((gBase.max - gBase.min), (gBase.N), order=FLAGS.order_type)

    # Add padding to both sides
    #print(f'gMinIn: {gMinIn.shape} padding {padding.shape}')
    gMinIn -= padding
    gMaxIn += padding

    # Initialize
    gMaxOut = zeros(gBase.dim, 1, dtype=np.float64, order=FLAGS.order_type)
    gMinOut = zeros(gBase.dim, 1, dtype=np.float64, order=FLAGS.order_type)
    NOut = zeros(gBase.dim, 1, order=FLAGS.order_type)

    for dim  in range(gBase.dim):
        # Arbitrary reference point
        refGridPt = gBase.min[dim]

        # Get minimum and maximum bounds for this dimension
        ptrMax = np.floor((gMaxIn[dim] - refGridPt) / dx[dim], order=FLAGS.order_type)
        gMaxOut[dim] = refGridPt + ptrMax*dx[dim]

        ptrMin = np.ceil((gMinIn[dim] - refGridPt) / dx[dim], order=FLAGS.order_type)
        gMinOut[dim] = refGridPt + ptrMin*dx[dim]

        # Get number of grid points
        NOut[dim] = int(ptrMax - ptrMin + 1)

    return gMinOut, gMaxOut, NOut
