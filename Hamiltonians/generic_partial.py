from Utilities import *

def genericPartial(t, data, derivMin, derivMax, schemeData, dim):
    g = schemeData.grid
    dynSys = schemeData.dynSys

    if isfield(dynSys, 'partialFunc'):
        alpha = dynSys.partialFunc(t, data, derivMin, derivMax, schemeData, dim)
        return alpha

    if not isfield(schemeData, 'uMode'):
        schemeData.uMode = 'min'

    if not isfield(schemeData, 'dMode'):
      schemeData.dMode = 'min'

    ## Compute control
    if isfield(schemeData, 'uIn'):
        # Control
        uU = schemeData.uIn
        uL = schemeData.uIn

    else:
        # Optimal control assuming maximum deriv
        uU = dynSys.get_opt_u(t, derivMax, schemeData.uMode, g.xs)

        # Optimal control assuming minimum deriv
        uL = dynSys.get_opt_u(t, derivMin, schemeData.uMode, g.xs)

    ## Compute disturbance
    if isfield(schemeData, 'dIn'):
        dU = schemeData.dIn
        dL = schemeData.dIn

    else:
        dU = dynSys.get_opt_v(t, derivMax, schemeData.dMode, g.xs)
        dL = dynSys.get_opt_v(t, derivMin, schemeData.dMode, g.xs)

    ## Compute alpha
    dxUU = dynSys.dynamics(t, schemeData.grid.xs, uU, dU)
    dxUL = dynSys.dynamics(t, schemeData.grid.xs, uU, dL)
    dxLL = dynSys.dynamics(t, schemeData.grid.xs, uL, dL)
    dxLU = dynSys.dynamics(t, schemeData.grid.xs, uL, dU)

    alpha = np.maximum(np.abs(dxUU[dim]), np.abs(dxUL[dim]))
    alpha = np.maximum(alpha, np.abs(dxLL[dim]))
    alpha = np.maximum(alpha, np.abs(dxLU[dim]))

    return alpha
