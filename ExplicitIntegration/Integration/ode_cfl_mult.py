from utils import *


def odeCFLmultipleSteps(intFunc, schemeFunc, tspan, y0, options, schemeData):
    """
    % odeCFLmultipleSteps: Handle the length(tspan) > 2 case for odeCFLn.
    %
    % [ t, y, schemeData ] = ...
    %      odeCFLmultipleSteps(intFunc, schemeFunc, tspan, y0, options, schemeData)
    %
    % Makes repeated calls to an odeCFLn routine to integrate a system forward
    %   in time, stopping at specified times to record the data.
    %
    % This routine is not intended to be called directly by the toolbox user;
    %   rather, it is some common code for the odeCFLn routines that was factored
    %   into a separate routine.
    %
    % Apart from the parameter intFunc, the parameters for this routine are
    %   the same as those for odeCFLn.
    %
    % parameters:
    %   intFunc      Function handle to an odeCFLn integration routine for
    %                  CFL constrained ODE systems.  This routine is used
    %                  to integrate between the pairs of times in tspan.
    %   schemeFunc	 Function handle to a CFL constrained ODE system
    %                  (typically an approximation to an HJ term, see below).
    %   tspan        Range of time over which to integrate (see below).
    %   y0           Initial condition vector
    %                  (typically the data array in vector form).
    %   options      An option structure generated by odeCFLset
    %                  (use [] as a placeholder if necessary).
    %   schemeData   Structure passed through to schemeFunc.
    %
    %
    %   t            Output time(s) (see below).
    %   y            Output state (see below).
    %   schemeData   Output version of schemeData (see below).
    %
    % The time interval tspan must be given as
    %   A vector with three or more entries, in which case the output will
    %   be column vector t = tspan and each row of y will be the solution
    %   at one of the times in tspan.  Unlike Matlab's ode suite routines,
    %   odeCFLmultipleSteps just repeatedly calls the function handle intfunc
    %   to accomplish this goal, so it is not particularly efficient.

    % Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
    % This software is used, copied and distributed under the licensing
    %   agreement contained in the file LICENSE in the top directory of
    %   the distribution.
    %
    % Factored from odeCFLn, Ian Mitchell, 12/06/04.
    # Lekan Aug 16, 2021
    """
    #---------------------------------------------------------------------------
    # Number of timesteps to be returned.
    numT = len(tspan)

    #---------------------------------------------------------------------------
    # If we were asked for the solution at multiple timesteps,
    #   call back for each pair of timesteps.
    if(numT > 2):
        t = tspan.reshape(numT, 1)

        if(iscell(y)):
            numY = len(y);
            y = cell(numY, 1);
            for i in range(numY):
                y[i] = zeros(numT, len(y0[i]));
                y[i][0,:] = y0[i].T;
        else:
            y = zeros(numT, length(y0));
            y[0,:] = y0.T;

        yout = y;
        for n in range(1, numT+1):
            t[n], yout, schemeData = intFunc(schemeFunc, np.hstack((t[n-1], t[n])), yout, schemeData, options)

            if(iscell(y)):
                for i in range(numY):
                    y[i][n,:] = yout[i].T
            else:
                y[n,:] = yout.T
    else:
        # This routine is only for finding the solution at multiple timesteps.
        error('tspan must contain at least three entries');

    return  t, y, schemeData
