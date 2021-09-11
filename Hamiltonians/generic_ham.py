from Utilities import *

def genericHam(t, data, deriv, schemeData):

    # Input unpacking
    dynSys = schemeData.dynSys

    if not isfield(schemeData, 'uMode'):
        schemeData.uMode = 'min'

    if not isfield(schemeData, 'dMode'):
        schemeData.dMode = 'max'

    if not isfield(schemeData, 'tMode'):
        schemeData.tMode = 'backward'

    # Custom derivative for MIE
    if isfield(schemeData, 'deriv'):
        deriv = schemeData.deriv

    ## Optimal control and disturbance
    if isfield(schemeData, 'uIn'):
        u = schemeData.uIn
    else:
        u = dynSys.get_opt_u(t, deriv, uMode=schemeData.uMode, y=schemeData.grid.xs)

    if isfield(schemeData, 'dIn'):
        d = schemeData.dIn
    else:
        d = dynSys.get_opt_v(t, deriv, dMode=schemeData.dMode, y=schemeData.grid.xs)

    hamValue = 0
    ## MIE
    if isfield(schemeData, 'side'):
        if strcmp(schemeData.side, 'lower'):
            TIderiv = -1
        elif strcmp(schemeData.side, 'upper'):
            TIderiv = 1
        else:
            error('Side of an MIE function must be upper or lower!')

    ## Plug optimal control into dynamics to compute Hamiltonian
    dx = dynSys.dynamics(t, schemeData.grid.xs, u, d)
    for i in range(dynSys.nx):
        hamValue += deriv[i]*dx[i]

    if isfield(dynSys, 'TIdim') and dynSys.TIdim:
        TIdx = dynSys.TIdyn(t, schemeData.grid.xs, u, d)
        hamValue += TIderiv*TIdx[0]

    ## Negate hamValue if backward reachable set
    if strcmp(schemeData.tMode, 'backward'):
        hamValue = -hamValue

    if isfield(schemeData, 'side'):
        if strcmp(schemeData.side, 'upper'):
            hamValue = -hamValue

    return hamValue
