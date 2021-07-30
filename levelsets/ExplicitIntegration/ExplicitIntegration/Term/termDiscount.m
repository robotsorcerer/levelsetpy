function [ ydot, stepBound, schemeData ] = termDiscount(t, y, schemeData)
% termDiscount: approximate a discounting term in an HJ PDE
%
% [ ydot, stepBound, schemeData ] = termDiscount(t, y, schemeData)
%
% Computes a discounting term \lambda(x) \phi(x) 
%   which depends directly on the value of the level set function
%   (no derivative is involved).
%
% This term is often used for time discounting in optimal control and
%   differential games, or as a "killing process" in SDEs.
%
% Note that \lambda(x) >= 0 is required, otherwise the resulting
%   HJ PDE will not be "proper" and the viscosity solution theory
%   will not apply (for example, the solution could become discontinuous).
%
% Because there is no spatial derivative in this term, there is no
%   corresponding CFL restriction (stepBound = +inf).  Consequently,
%   this term should always be combined with some term involving a
%   spatial derivative.
%
% parameters:
%   t            Time at beginning of timestep.
%   y            Data array in vector form.
%   schemeData	 A structure (see below).
%
%   ydot	 Change in the data array, in vector form.
%   stepBound	 CFL bound on timestep for stability.
%                  Always returned as stepBound = +inf.
%   schemeData   The same as the input argument (unmodified).
%
% schemeData is a structure containing data specific to this type of 
%   term approximation.  For this function it contains the field(s)
%
%   .grid	 Grid structure (see processGrid.m for details).
%   .lambda      A description of the discount factor (see below).
%
% It may contain additional fields at the user's discretion.
%
% schemeData.lambda can provide the discount factor in one of two ways:
%   1) For time invariant discount, a scalar or an array the same 
%      size as data.
%   2) For general discount, a function handle to a function with prototype
%      lambda = scalarGridFunc(t, data, schemeData), where the output 
%      lambda is the scalar/array from (1) and the input arguments are 
%      the same as those of this function (except that data = y has been 
%      reshaped to its original size).  In this case, it may be useful to 
%      include additional fields in schemeData.
%
% For evolving vector level sets, y may be a cell vector.  If y is a cell
%   vector, schemeData may be a cell vector of equal length.  In this case
%   all the elements of y (and schemeData if necessary) are ignored except
%   the first.  As a consequence, if schemeData.lambda is a function handle
%   the call to scalarGridFunc will be performed with a regular data array
%   and a single schemeData structure (as if no vector level set was present).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/18/04
% Updated to handle vector level sets.  Ian Mitchell 11/23/04.

  %---------------------------------------------------------------------------
  % For vector level sets, ignore all the other elements.
  if(iscell(schemeData))
    thisSchemeData = schemeData{1};
  else
    thisSchemeData = schemeData;
  end

  checkStructureFields(thisSchemeData, 'grid', 'lambda');

  grid = thisSchemeData.grid;

  %---------------------------------------------------------------------------
  if(iscell(y))
    data = reshape(y{1}, grid.shape);    
  else
    data = reshape(y, grid.shape);
  end

  %---------------------------------------------------------------------------
  % Get discount factor.
  if(isa(thisSchemeData.lambda, 'double'))
    lambda = thisSchemeData.lambda;
  elseif(isa(thisSchemeData.lambda, 'function_handle'))
    data = reshape(y, thisSchemeData.grid.shape);
    lambda = feval(thisSchemeData.lambda, t, data, thisSchemeData);
  else
    error('schemeData.lambda must be a scalar, array or function handle');
  end
  
  %---------------------------------------------------------------------------
  % Compute the update (including negation for RHS of ODE).
  delta = lambda .* data;
  ydot = -delta(:);

  % No derivative, so no timestep limit.
  stepBound = +inf;
