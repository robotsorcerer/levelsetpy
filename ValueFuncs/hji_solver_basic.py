import copy
import numpy as np
from os.path import join
from datetime import datetime
from Hamiltonians import *
from ExplicitIntegration import odeCFL3, odeCFLset, termLaxFriedrichs
from SpatialDerivative import *
from Utilities import *
from ValueFuncs import *
import matplotlib.pyplot as plt


def HJI_PDE_Solver(data0, tau, schemeData, extraArgs):
    """
        Basic HJI Solver. Adopted from Sylvia's code.

        Lekan Molu, 09/21/2021
    """
    #SchemeData
    small = 1e-4
    lowMemory = False
    compMethod=None
    stopConverge = False
    stopSet = False
    keepLast = False if not isfield(extraArgs, 'keepLast') else extraArgs.keepLast
    ## Numerical approximation functions

    # schemeData.dissFunc, schemeData.derivFunc = artificialDissipationGLF, upwindFirstENO3

    ## Time integration
    ode_cfl_opts = Bundle({'factorCFL': 0.8, 'singleStep': 'on'})
    integratorOptions = odeCFLset(ode_cfl_opts)

    startTime = cputime()

    ## Initialize PDE solution
    data0size = size(data0)
    g = schemeData.grid
    # print(f'data0size: {data0size}, g.dim: {g.dim}')
    if data0.ndim == g.dim:
        data = np.zeros((data0size[0:g.dim] +(len(tau), )), dtype=np.float64, order=ORDER_TYPE)
        # print(f'data: {data.shape}, tau: {len(tau)}, datao: {data0.shape}')
        data[:,:,:, 0] = data0

        istart = 1
    elif data0.ndim == g.dim + 1:
        # Continue an old computation
        if keepLast:
            data = data0[:,:,:,data0size[-1]]
        elif lowMemory:
            data = data0[:,:,:,data0size[-1]].astype(np.float32)
        else:
            data = np.zeros((data0size[:gDim].shape+(len(tau),)), dtype=np.float64)
            data[:,:,:,data0size[-1]] = data0


        # Start at custom starting index if specified
        if isfield(extraArgs, 'istart'):
            istart = extraArgs.istart
        else:
            istart = data0size[-1]+1

    else:
        error('Inconsistent initial condition dimension!')

    if isfield(extraArgs,'ignoreBoundary') and extraArgs.ignoreBoundary:
        _, dataTrimmed = truncateGrid(g, data0, g.min+4*g.dx, g.max-4*g.dx)

    for i in range(istart, len(tau)):
        if keepLast:
            y0 = data
        elif lowMemory:
            if flipOutput:
                y0 = data[:,:,:, 0]
            else:
                y0 = data[:,:,:,size(data, g.dim+1)]
        else:
            y0 = data[:,:,:,i-1]
        # print(f'data: {data.shape}, data[:,:,:,{i-1}]: {data[:,:,:,i-1].shape}, y0: {y0.shape}')

        y = expand(y0.flatten(order=ORDER_TYPE), 1)
        # print(f'y: {y.shape},  y0: {y0.shape}')

        tNow = tau[i-1]

        ## Main integration loop to get to the next tau(i)
        while tNow < tau[i] - small:
            # Save previous data if needed
            if compMethod =='minVOverTime' or compMethod =='maxVOverTime':
                yLast = copy.deepcopy(y)
                debug(f'Solving HJ Value at Time Step: {tNow:.4f}/{tau[i]:.2f}')
            """
                Solve hamiltonian and apply to value function (y) to get updated
                value function. The solution to the conservation equation is the
                same as the derivative of the solution to the Hamiltonian.
            """
            tNow, y, _ = odeCFL3(termLaxFriedrichs, [tNow, tau[i]], y, integratorOptions, schemeData)

            if np.any(np.isnan(y)):
                error(f'Nans encountered in the integrated result of HJI PDE data')

            # "Mask" using obstacles
            if  isfield(extraArgs, 'obstacleFunction'):
                if strcmp(obsMode, 'time-varying'):
                    obstacle_i = obstacles[i,...]

                y = omax(y, -obstacle_i)

            # Update target function
            if isfield(extraArgs, 'targetFunction'):
                if strcmp(targMode, 'time-varying'):
                    target_i = targets[i,...]

        # Reshape value function
        data_i = y.reshape(g.shape, order=ORDER_TYPE))
        if keepLast:
            data = data_i
        elif lowMemory:
            if flipOutput:
                data = np.concatenate((y.reshape(g.shape, order=ORDER_TYPE), data), g.dim+1)
            else:
                data = np.concatenate((data, y.reshape(g.shape, order=ORDER_TYPE)), g.dim+1)
        else:
            # print(f'data_i: {data_i.shape}, data: {data.shape}')
            # I = [slice(None)]*data.ndim; I[-1] = i
            data[:,:,:,i] = data_i


        # If we're stopping once converged, print how much change there was in
        # the last iteration
        if stopConverge:
            if  isfield(extraArgs,'ignoreBoundary')  and extraArgs.ignoreBoundary:
                _ , dataNew = truncateGrid(g, data_i, g.min+4*g.dx, g.max-4*g.dx)
                change = np.max(np.abs(dataNew.flatten(order=ORDER_TYPE)- \
                            dataTrimmed.flatten(order=ORDER_TYPE), order=ORDER_TYPE))
                dataTrimmed = dataNew
                if not quiet:
                    info(f'Max change since last iteration: {change}')

            else:
                # check this
                change = np.max(np.abs(y - expand(y0.flatten(order=ORDER_TYPE)), 1)))
                if not quiet:
                    info(f'Max change since last iteration: {change}')

        ## If commanded, stop the reachable set computation once it contains
        # the initial state.
        if isfield(extraArgs, 'stopInit'):
            initValue = eval_u(g, data_i, extraArgs.stopInit)
            if not np.isnan(initValue) and initValue <= 0:
                extraOuts.stoptau = tau[i]
                tau  = tau[:i+1]

                if not lowMemory and not keepLast:
                    data = data[:i+1, ...]

                break

        ## Stop computation if reachable set contains a "stopSet"
        if stopSet:
            dataInds = np.nonzero(data_i <= stopLevel)

            if isfield(extraArgs, 'stopSetInclude'):
                stopSetFun = np.all
            else:
                stopSetFun = np.any


            if stopSetFun(np.isin(setInds, dataInds)):
                extraOuts.stoptau = tau[i]
                tau  = tau[:i+1]

                if not lowMemory and not keepLast:
                    data = data[:i+1, ...]

                break

        ## Stop computation if we've converged
        if stopConverge and change < convergeThreshold:

            if isfield(extraArgs, 'discountFactor') and \
                extraArgs.discountFactor and \
                isfield(extraArgs, 'discountAnneal') and \
                extraArgs.discountFactor != 1:

                if strcmp(extraArgs.discountAnneal, 'soft'):
                    extraArgs.discountFactor = 1-((1-extraArgs.discountFactor)/2)

                    if np.abs(1-extraArgs.discountFactor) < .00005:
                        extraArgs.discountFactor = 1

                    info(f'Discount factor: {extraArgs.discountFactor}')
                elif strcmp(extraArgs.discountAnneal, 'hard') or extraArgs.discountAnneal==1:
                    extraArgs.discountFactor = 1
                    info(f'Discount factor: {extraArgs.discountFactor}')

            else:
                extraOuts.stoptau = tau[i]
                tau  = tau[:i+1]

                if not lowMemory and not keepLast:
                    data = data[:i+1, ...]

                break

        ## If commanded, visualize the level set.
        if (isfield(extraArgs, 'visualize')  and (isinstance(extraArgs.visualize, Bundle) or extraArgs.visualize == 1)) or (isfield(extraArgs, 'makeVideo') and extraArgs.makeVideo):
            timeCount += 1

            # Number of dimensions to be plotted and to be projected
            pDims = np.count_nonzero(plotDims)
            # print(f'projpt: {projpt}, {len(projpt)}')
            if isnumeric(projpt):
                projDims = len(projpt)
            else:
                projDims = gDim - pDims

            # Basic Checks
            if(len(plotDims) != gDim or projDims != (gDim - pDims)):
                error('Mismatch between plot and grid dimensions!')

            #---Perform Projections--------------------------------------------
            if projDims == 0:
                gPlot = g
                dataPlot = data_i

                if strcmp(obsMode, 'time-varying'):
                    obsPlot = copy.copy(obstacle_i)

                if strcmp(targMode, 'time-varying'):
                    targPlot = copy.copy(target_i)

            else:
                # if projpt is a cell, project each dimensions separately. This
                # allows us to take the union/intersection through some dimensions
                # and to project at a particular slice through other dimensions.
                if iscell(projpt) and len(projpt)>1:
                    idx = np.nonzero(plotDims==0)
                    # print('plotDims: ', plotDims)
                    plotDimsTemp = np.ones((plotDims))
                    gPlot = g
                    dataPlot = data_i
                    if strcmp(obsMode, 'time-varying'):
                        obsPlot = obstacle_i


                    if strcmp(targMode, 'time-varying'):
                        targPlot = target_i


                    for ii in range(len(idx)-1, -1, -1):
                        plotDimsTemp[idx[ii]] = 0
                        if strcmp(obsMode, 'time-varying'):
                            _ , obsPlot = proj(gPlot, obsPlot, np.logical_not(plotDimsTemp),\
                                projpt[ii])


                        if strcmp(targMode, 'time-varying'):
                            _ , targPlot = proj(gPlot, targPlot, np.logical_not(plotDimsTemp),\
                                projpt[ii])


                        gPlot, dataPlot = proj(gPlot, dataPlot, np.logical_not(plotDimsTemp), projpt[ii])
                        plotDimsTemp = ones(1,gPlot.dim)


                else:
                    gPlot, dataPlot = proj(g, data_i, np.logical_not(plotDims).astype(np.int64), projpt)

                    if strcmp(obsMode, 'time-varying'):
                        _ , obsPlot = proj(g, obstacle_i, np.logical_not(plotDims).astype(np.int64), projpt)


                    if strcmp(targMode, 'time-varying'):
                        _ , targPlot = proj(g, obstacle_i, np.logical_not(plotDims).astype(np.int64), projpt)

    return data, tau
