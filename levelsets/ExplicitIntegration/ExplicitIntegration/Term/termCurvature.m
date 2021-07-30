function [ ydot, stepBound, schemeData ] = termCurvature(t, y, schemeData)
% termCurvature: approximate a motion by mean curvature term in an HJ PDE.
%
% [ ydot, stepBound, schemeData ] = termCurvature(t, y, schemeData)
%
% Computes an approximation of motion by mean curvature for a
%   Hamilton-Jacobi PDE.  This is a second order equation that simplifies to
%   a heat equation if the function is a signed distance function.
%   Specifically:
%
%                 D_t \phi - b(x,t) \kappa(x) \| \grad \phi \| = 0.
%
%   where \kappa(x) is the mean curvature.
%
% Based on methods outlined in O&F, chapters 4.1 & 4.2.
%
% parameters:
%   t            Time at beginning of timestep.
%   y            Data array in vector form.
%   schemeData	 A structure (see below).
%
%   ydot	 Change in the data array, in vector form.
%   stepBound	 CFL bound on timestep for stability.
%   schemeData   The same as the input argument (unmodified).
%
% schemeData is a structure containing data specific to this type of 
%   term approximation.  For this function it contains the field(s)
%
%   .grid	   Grid structure (see processGrid.m for details).
%   .curvatureFunc Function handle to finite difference curvature approx.
%                    Should provide both curvature and gradient magnitude.
%   .b		   Multiplier (should be non-negative for well-posedness);
%                    b may be a scalar constant or an array of size(data).
%   .passVLS       An optional boolean used for vector level sets (see below).
%                    Default is 0 (ie ignore vector level sets).
%
% It may contain additional fields at the user's discretion.
%
% schemeData.b can provide the multiplier in one of two ways:
%   1) For time invariant multipliers, a scalar or an array the same 
%      size as data.
%   2) For general multipliers, a function handle to a function with prototype
%      b = scalarGridFunc(t, data, schemeData), where the output b is the
%      scalar/array from (1) and the input arguments are the same as those
%      of this function (except that data = y has been reshaped to its
%      original size).  In this case, it may be useful to include additional
%      fields in schemeData.  
%
% For evolving vector level sets, y may be a cell vector.  If y is a cell
%   vector, schemeData may be a cell vector of equal length.  In this case
%   all the elements of y (and schemeData if necessary) are ignored except
%   the first.  As a consequence, if schemeData.b is a function handle
%   the call to scalarGridFunc will be performed with a regular data array
%   and a single schemeData structure (as if no vector level set was present).
%
% This default behavior of ignoring the vector level set in the call
%   to scalarGridFunc may be overridden by setting schemeData.passVLS = 1.
%   In this case the data argument (and schemeData argument, if necessary) 
%   in the call to velocityFunc will be the full cell vectors.  The current
%   data array (and schemeData structure, if necessary) will be the first
%   element of these cell vectors.  In order to properly reshape the other 
%   elements of y, the corresponding schemeData structures must contain 
%   an appropriate grid structure.
%
% In the notation of OF text,
%
%   data = y	  \phi
%   curvatureFunc function to calculate \kappa and |\grad \phi|
%   b		  b
%
%   delta = ydot  +b \kappa |\grad \phi|


% Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell 6/3/03
% Calling parameters significantly modified, Ian Mitchell 2/13/04.
% Updated to handle vector level sets.  Ian Mitchell 11/23/04.
% Updated to include passVLS option.  Ian Mitchell 02/14/05.

  %---------------------------------------------------------------------------
  if(iscell(schemeData))
    thisSchemeData = schemeData{1};
  else
    thisSchemeData = schemeData;
  end

  checkStructureFields(thisSchemeData, 'grid', 'b', 'curvatureFunc');

  grid = thisSchemeData.grid;

  %---------------------------------------------------------------------------
  if(iscell(y))
    data = reshape(y{1}, grid.shape);    
  else
    data = reshape(y, grid.shape);
  end

  %---------------------------------------------------------------------------
  % Get multiplier
  if(isa(thisSchemeData.b, 'double'))
    b = thisSchemeData.b;

  elseif(isa(thisSchemeData.b, 'function_handle'))

    if(iscell(y))
      % If there is a vector level set.

      if(isfield(thisSchemeData, 'passVLS') && thisSchemeData.passVLS)
        % Pass the vector level set information through.
        numY = length(y);
        dataV = cell(numY, 1);
        for i = 1 : numY
          if(iscell(schemeData))
            dataV{i} = reshape(y{i}, schemeData{i}.grid.shape);
          else
            dataV{i} = reshape(y{i}, schemeData.grid.shape);
          end
        end
        b = feval(thisSchemeData.b, t, dataV, schemeData);

      else
        % Ignore any vector level set.
        b = feval(thisSchemeData.b, t, data, thisSchemeData);
      end

    else
      % There is no vector level set.
      b = feval(thisSchemeData.b, t, data, thisSchemeData);
    end


  else
    error('schemeData.b must be a scalar, array or function handle');

  end

  %---------------------------------------------------------------------------
  % According to O&F equation (4.5).
  [ curvature, gradMag ] = feval(thisSchemeData.curvatureFunc, grid, data);
  delta = -b .* curvature .* gradMag;
  
  %---------------------------------------------------------------------------
  % According to O&F equation (4.7).
  stepBound = 1 / (2 * max(b(:)) * sum(grid.dx .^ -2));
  
  % Reshape output into vector format and negate for RHS of ODE.
  ydot = -delta(:);
