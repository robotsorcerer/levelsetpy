function [ value, schemeDataOut ] = ...
                     terminalEventConverge(t, y, tOld, yOld, schemeDataIn)
% terminalEventConverge: Detects convergence of the integration.
%
%  [ value, schemeDataOut ] = ...
%                    terminalEventConverge(t, y, tOld, yOld, schemeDataIn)
%
% This routine implements the terminalEvent prototype for a common
%   operation: detecting that the implicit surface function is no
%   longer changing; in other words, that the calculation has converged.
%
% Convergence is detected by checking whether the norm of the update made 
%   to the implicit surface function falls below some tolerance.
%
% Notes about norm choices.  Difference y - yOld is measured relative to y.
%
% Doesn't work on vector level sets.
%
% The schemeData structure is not modified by this function.
%
% Parameters:
%   t              Current time.
%   y              The current implicit surface function.
%   tOld           Time at the last timestep.
%   yOld           Implicit surface function during the last timestep.
%   schemeDataIn   Input version of a structure (see below).
%
%   value          A scalar indicating the degree to which
%                    the data was updated.
%   schemeDataOut  Output version of the structure (unmodified).
%
% schemeData is a structure containing data specific to this terminal
%   event.  For this function it contains the field(s)
%
%   .convergeAbsTol  Absolute tolerance for convergence (scalar).
%                      Default is 1e-6.
%   .convergeRelTol  Relative tolerance for convergence (scalar).
%                      Default is 1e-3.
%   .convergeNorm    Norm in which convergence is measured
%                      In order of increasing tightness, the options are:
%                      'average'    Average over all node points (default).
%                      'maximum'    Maximum over all node points.
%                      'pointwise'  Each node point tested individually.
%
% schemeData may contain other fields.

% Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell 1/30/05

  if iscell(y)
    error('TerminalEventConverge does not work on vector level sets.');
  end
    
  % schemeData is unchanged.
  schemeDataOut = schemeDataIn;

  if(isfield(schemeDataIn, 'convergeAbsTol'))
    absTol = schemeDataIn.convergeAbsTol;
  else
    absTol = 1e-6;
  end

  if(isfield(schemeDataIn, 'convergeRelTol'))
    relTol = schemeDataIn.convergeRelTol;
  else
    relTol = 1e-3;
  end

  if(isfield(schemeDataIn, 'convergeNorm'))
    convergeNorm = schemeDataIn.convergeNorm;
  else
    convergeNorm = 'average';
  end

  % Compute the tolerance.
  tol = max(relTol * findNorm(y, convergeNorm), absTol);

  % Compute the norm of the update.
  update = findNorm(y - yOld, convergeNorm);

  %[ max(tol(:)), max(update(:) - tol(:)) ]

  % Find the value for the most recent timestep.
  %   Since magnitude of value is irrelevant, just return +-1.
  if(all(update < tol))
    value = -1;
  else
    value = +1;
  end



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function normX = findNorm(x, normType)
% findNorm: Computes one of several norms for a grid function.
%
%   normX = findNorm(x, normType)
%
% Three norms are available:
%
%    'maximum'    returns maximum absolute value of all nodes in x (scalar).
%    'average'    returns average absolute value of all nodes in x (scalar).
%    'pointwise'  returns absolute value of all nodes in x (same size as x).
%
% Parameters:
%   x              Array over which the norm is to be taken.
%   normType       One of the strings listed above.
%
%   normX          The norm of the array as measured in the appropriate norm.

switch(normType)

 case 'average'
  normX = mean(abs(x(:)));

 case 'maximum'
  normX = max(abs(x(:)));

 case 'pointwise'
  normX = abs(x);

 otherwise
  error([ 'Unknown type of norm: ' normType ]);

end
