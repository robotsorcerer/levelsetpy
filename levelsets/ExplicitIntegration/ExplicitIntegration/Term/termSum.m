function [ ydot, stepBound, schemeData ] = termSum(t, y, schemeData)
% termSum: Combine a collection of spatial HJ term approximations.
%
% [ ydot, stepBound, schemeData ] = termSum(t, y, schemeData)
%
% This function independently evaluates a collection of HJ term 
%   approximations and returns their elementwise sum.
%
% Note that the HJ term approximations must be completely independent
%   of one another.
%
% The CFL restrictions are inverse summed:
%	stepBound_sum = (\sum_i (1 / stepBound_i))^-1
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
%   .innerFunc   A cell vector of function handles to the term 
%                  approximations that will be summed.
%   .innerData   A cell vector of schemeData structures to pass to 
%                  each of the innerFunc elements.  This vector
%                  must be the same size as the vector innerFunc.
%
% It may contain addition fields at the user's discretion.
%
% While termSum will not change the schemeData structure,
%   the schemeData.innerData components may be changed during the calls
%   to the schemeData.innerFunc function handles.
%
% For evolving vector level sets, y may be a cell vector.  In this case
%   the entire y cell vector is passed unchanged in the calls to the 
%   schemeData.innerFunc function handles.
%
% If y is a cell vector, schemeData may be a cell vector of equal length.  
%   In this case, schemeData{1} contains the fields listed above.  In the
%   calls to the schemeData{1}.innerFunc function handles, the schemeData 
%   cell vector is passed unchanged except that the element schemeData{1} 
%   is replaced with the corresponding element of the 
%   schemeData{1}.innerData cell vector.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 4/9/04
% Updated to handle vector level sets.  Ian Mitchell 11/23/04

  %---------------------------------------------------------------------------
  % For vector level sets, get the first element.
  if(iscell(schemeData))
    thisSchemeData = schemeData{1};
  else
    thisSchemeData = schemeData;
  end

  checkStructureFields(thisSchemeData, 'innerFunc', 'innerData');

  %---------------------------------------------------------------------------
  % Check that innerFunc and innerData are the same size cell vectors.
  if(~iscell(thisSchemeData.innerFunc) | ~iscell(thisSchemeData.innerData))
    error('schemeData.innerFunc and schemeData.innerData %s', ...
          'must be cell vectors');
  end

  numSchemes = length(thisSchemeData.innerFunc(:));

  if(numSchemes ~= length(thisSchemeData.innerData(:)))
    error('schemeData.innerFunc and schemeData.innerData must be %s', ...
          'the same length');
  end

  %---------------------------------------------------------------------------
  % Calculate sum of updates (inverse sum of stepBounds).
  ydot = 0;
  stepBoundInv = 0;
  for i = 1 : numSchemes
    
    % Extract the appropriate inner data structure.
    if(iscell(schemeData))
      innerData = schemeData;
      innerData{1} = schemeData{1}.innerData{i};
    else
      innerData = schemeData.innerData{i};
    end

    % Compute this component of the update.
    [ updateI, stepBoundI, innerData ] = ...
                           feval(thisSchemeData.innerFunc{i}, t, y, innerData);
    ydot = ydot + updateI;
    stepBoundInv = stepBoundInv + 1 / stepBoundI;

    % Store any modifications of the inner data structure.
    if(iscell(schemeData))
      schemeData{1}.innerData{i} = innerData{1};
    else
      schemeData.innerData{i} = innerData;
    end

  end

  %---------------------------------------------------------------------------
  % Final timestep bound.
  if(stepBoundInv == 0)
    stepBound = inf;
  else
    stepBound = 1 / stepBoundInv;
  end
