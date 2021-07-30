function [ ydot, stepBound, schemeData ] = termRestrictUpdate(t, y, schemeData)
% termRestrictUpdate: restrict the sign of a term to be positive or negative.
%
% [ ydot, stepBound, schemeData ] = termRestrictUpdate(t, y, schemeData)
%
% Given some other HJ term approximation, this function either restricts
%   the sign of that update to be either:
%       nonnegative (level set function only decreases), or
%       nonpositive (level set function only increases).
%
% The CFL restriction of the other HJ term approximation is returned 
%   without modification.
%
% parameters:
%   t            Time at beginning of timestep.
%   y            Data array in vector form.
%   schemeData	 A structure (see below).
%
%   ydot	 Change in the data array, in vector form.
%   stepBound	 CFL bound on timestep for stability.
%   schemeData   A structure (see below).
%
% schemeData is a structure containing data specific to this type of 
%   term approximation.  For this function it contains the field(s)
%
%   .innerFunc   Function handle for the approximation scheme to
%                  calculate the unrestricted HJ term.
%   .innerData   schemeData structure to pass to innerFunc.
%   .positive    Boolean, true if update must be positive (nonnegative)
%                  (optional, default = 1).
%
% It may contain addition fields at the user's discretion.
%
% While termRestrictUpdate will not change the schemeData structure,
%   the schemeData.innerData component may be changed during the call
%   to the schemeData.innerFunc function handle.
%
% For evolving vector level sets, y may be a cell vector.  In this case
%   the entire y cell vector is passed unchanged in the call to the 
%   schemeData.innerFunc function handle.
%
% If y is a cell vector, schemeData may be a cell vector of equal length.  
%   In this case, schemeData{1} contains the fields listed above.  In the
%   call to schemeData{1}.innerFunc, the schemeData cell vector is passed
%   unchanged except that the element schemeData{1} is replaced with 
%   the contents of the schemeData{1}.innerData field.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 3/26/04
% Updated to handle vector level sets.  Ian Mitchell 11/23/04.

  %---------------------------------------------------------------------------
  % For vector level sets, get the first element.
  if(iscell(schemeData))
    thisSchemeData = schemeData{1};
  else
    thisSchemeData = schemeData;
  end

  checkStructureFields(thisSchemeData, 'innerFunc', 'innerData');

  %---------------------------------------------------------------------------
  % Extract the appropriate inner data structure.
  if(iscell(schemeData))
    innerData = schemeData;
    innerData{1} = schemeData{1}.innerData;
  else
    innerData = schemeData.innerData;
  end

  %---------------------------------------------------------------------------
  % Get the unrestricted update.
  [ unRestricted, stepBound, innerData ] = ...
                              feval(thisSchemeData.innerFunc, t, y, innerData);

  %---------------------------------------------------------------------------
  % Store any modifications of the inner data structure.
  if(iscell(schemeData))
    schemeData{1}.innerData = innerData{1};
  else
    schemeData.innerData = innerData;
  end

  %---------------------------------------------------------------------------
  % Default to positive (nonnegative) update restriction.
  if(isfield(thisSchemeData, 'positive'))
    positive = thisSchemeData.positive;
  else
    positive = 1;
  end

  %---------------------------------------------------------------------------
  % Restrict the update (stepBound is returned unchanged).  
  %   Do not negate for RHS of ODE (that is handled by innerFunc).
  if(positive)
    ydot = max(unRestricted, 0);
  else
    ydot = min(unRestricted, 0);
  end
