from utils import *

def genericPartial(t, data, derivMin, derivMax, schemeData, dim):
    g = schemeData.grid
    dynSys = schemeData.dynSys

    if callable(dynSys, 'partialFunc'):
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
        uU = dynSys.get_opt_u(t, g.xs, derivMax, schemeData.uMode)

        # Optimal control assuming minimum deriv
        uL = dynSys.get_opt_u(t, g.xs, derivMin, schemeData.uMode)

    ## Compute disturbance
    if isfield(schemeData, 'dIn'):
        dU = schemeData.dIn
        dL = schemeData.dIn

    else:
        dU = dynSys.get_opt_v(t, g.xs, derivMax, schemeData.dMode)
        dL = dynSys.get_opt_v(t, g.xs, derivMin, schemeData.dMode)

    ## Compute alpha
    dxUU = dynSys.update_dynamics(schemeData.grid.xs, uU, dU, t=t)
    dxUL = dynSys.update_dynamics(schemeData.grid.xs, uU, dL, t=t)
    dxLL = dynSys.update_dynamics(schemeData.grid.xs, uL, dL, t=t)
    dxLU = dynSys.update_dynamics(schemeData.grid.xs, uL, dU, t=t)

    alpha = omax(np.abs(dxUU[dim]), np.abs(dxUL[dim]))
    alpha = omax(alpha, np.abs(dxLL[dim]))
    alpha = omax(alpha, np.abs(dxLU[dim]))

    return alpha
