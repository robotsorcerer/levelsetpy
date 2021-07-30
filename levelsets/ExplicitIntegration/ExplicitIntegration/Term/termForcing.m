function [ ydot, stepBound, schemeData ] = termForcing(t, y, schemeData)
% termForcing: approximate a forcing term in an HJ PDE.
%
% [ ydot, stepBound, schemeData ] = termForcing(t, y, schemeData)
%
% Computes a forcing term F(x,t) for an HJ PDE.  This term is a catchall
%   for any part of the HJ PDE that is independent of the derivatives of 
%   the level set function \phi(x,t).
%
% This term may also depend on \phi itself: F(x, t, \phi).  In that case,
%   it must satisfy the monotonicity requirement:
%
%              F(x,t,r) \leq F(x,t,s)  for r \leq s and all x,t
%
%   otherwise the resulting HJ PDE will not be proper and viscosity solution
%   theory may not apply (for example, the solution may become discontinuous).
%
% For vector level set functions, terms dependent on other elements of 
%   the vector will fall into this category.  However, users are warned
%   that the existence, uniqueness and stability of systems of HJ PDEs
%   (ie vector level sets) is poorly characterized at present.
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
%   .forcing     A description of the forcing term (see below).
%   .passVLS     An optional boolean used for vector level sets (see below).
%                  Default is 0 (ie ignore vector level sets).
%
% It may contain additional fields at the user's discretion.
%
% schemeData.forcing can provide the forcing term in one of two ways:
%   1) For time invariant forcing term, a scalar or an array the same 
%      size as data.
%   2) For general forcing, a function handle to a function with prototype
%      F = scalarGridFunc(t, data, schemeData), where the output 
%      F is the scalar/array from (1) and the input arguments are 
%      the same as those of this function (except that data = y has been 
%      reshaped to its original size).  In this case, it may be useful to 
%      include additional fields in schemeData.
%
% For evolving vector level sets, y may be a cell vector.  If y is a cell
%   vector, schemeData may be a cell vector of equal length.  In this case
%   all the elements of y (and schemeData if necessary) are ignored except
%   the first.  As a consequence, if schemeData.forcing is a function handle
%   the call to scalarGridFunc will be performed with a regular data array
%   and a single schemeData structure (as if no vector level set was present).
%
% This default behavior of ignoring the vector level set in the call
%   to scalarGridFunc may be overridden by setting schemeData.passVLS = 1.
%   In this case the data argument (and schemeData argument, if necessary) 
%   in the call to scalarGridFunc will be the full cell vectors.  The current
%   data array (and schemeData structure, if necessary) will be the first
%   element of these cell vectors.  In order to properly reshape the other 
%   elements of y, the corresponding schemeData structures must contain 
%   an appropriate grid structure.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 9/03/04
% Updated to handle vector level sets.  Ian Mitchell 11/23/04.

  %---------------------------------------------------------------------------
  % For vector level sets, ignore all the other elements.
  if(iscell(schemeData))
    thisSchemeData = schemeData{1};
  else
    thisSchemeData = schemeData;
  end

  checkStructureFields(thisSchemeData, 'grid', 'forcing');

  %---------------------------------------------------------------------------
  % Get forcing function.
  if(isa(thisSchemeData.forcing, 'double'))
    forcing = thisSchemeData.forcing;

  elseif(isa(thisSchemeData.forcing, 'function_handle'))

    if(iscell(y))

      if(isfield(thisSchemeData, 'passVLS') && thisSchemeData.passVLS)
        % Pass the vector level set information through.
        numY = length(y);
        data = cell(numY, 1);
        for i = 1 : numY
          if(iscell(schemeData))
            data{i} = reshape(y{i}, schemeData{i}.grid.shape);
          else
            data{i} = reshape(y{i}, schemeData.grid.shape);
          end
        end
        forcing = feval(thisSchemeData.forcing, t, data, schemeData);

      else
        % Ignore any vector level set.
        data = reshape(y{1}, thisSchemeData.grid.shape);
        forcing = feval(thisSchemeData.forcing, t, data, thisSchemeData);
      end

    else
      % There is no vector level set.
      data = reshape(y, thisSchemeData.grid.shape);
      forcing = feval(thisSchemeData.forcing, t, data, thisSchemeData);
    end

  else
    error('schemeData.forcing must be a scalar, array or function handle');
  end

  %---------------------------------------------------------------------------
  % Compute the update (including negation for RHS of ODE).
  ydot = -forcing(:);

  % No derivative, so no timestep limit.
  stepBound = +inf;
