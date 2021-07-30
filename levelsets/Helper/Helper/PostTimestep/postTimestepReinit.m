function [ yOut, schemeDataOut ] = postTimestepReinit(t, yIn, schemeDataIn)
% postTimestepReinit: postTimestep routine to perform some reinitialization.
%
%  [ yOut, schemeDataOut ] = postTimestepReinit(t, yIn, schemeDataIn)
%
% This routine implements the postTimestepFunc prototype for a common
%   operation: reinitializing the level set function using the
%   reinitialization PDE (no need to find explicit surface representation).
%
% Reinitialization is accomplished by calling signedDistanceIterative.
%
% Parameters:
%   t              Current time.
%   yIn            Input version of the level set function, in vector form.
%   schemeDataIn   Input version of a structure (see below).
%
%   yOut           Output version of the level set function, in vector form.
%   schemeDataOut  Output version of the structure (unmodified).
%
% schemeData is a structure containing data specific to this type of 
%   term approximation.  For this function it contains the field(s)
%
%   .grid            Grid structure.
%   .reinitAccuracy  Controls the order of approximation.
%                      'low'         Use odeCFL1 and upwindFirstFirst.
%                      'medium'      Use odeCFL2 and upwindFirstENO2 (default).
%                      'high'        Use odeCFL3 and upwindFirstENO3.
%                      'veryHigh'    Use odeCFL3 and upwindFirstWENO5.
%   .reinitSteps     How many steps of reinitialization to perform.
%                      May be inf, in which case reinit to convergence.
%                      Defaults to five.
%   .reinitErrorMax  If the average update of nodes drops below
%                      errorMax * max(grid.dx), then assume that 
%                      reinitialization has converged and return early.
%                      This is therefore a relative error measurement.
%                      Defaults to 1e-3.
%   .reinitPerform   Boolean specifying whether reinitialization should
%                      be performed at all.  Useful to turn off reinit
%                      for selected elements of a vector level set.
%                      Defaults to 1 (ie perform reinitialization).
%
% schemeData may contain other fields.

% Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell 12/06/04.
% Modified to accept vector level sets, Ian Mitchell 2/16/05
% $Date: 2010-08-09 21:31:46 -0700 (Mon, 09 Aug 2010) $
% $Id: postTimestepReinit.m 50 2010-08-10 04:31:46Z mitchell $

  %---------------------------------------------------------------------------
  % Copy schemeData structure unchanged.
  schemeDataOut = schemeDataIn;

  %---------------------------------------------------------------------------
  % If this is a vector level set, operate on each element separately.
  if(iscell(yIn))
    yOut = cell(length(yIn), 1);
    for i = 1 : length(yIn)
      if(iscell(schemeDataIn))
        yOut{i} = doReinit(yIn{i}, schemeDataIn{i});
      else
        yOut{i} = doReinit(yIn{i}, schemeDataIn);
      end
    end
  else
    yOut = doReinit(yIn, schemeDataIn);
  end

  

%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function yOut = doReinit(yIn, schemeData)
% doReinit: Calls signedDistanceIterative to reinitialize a level set function.
%
%    yOut = doReinit(yIn, schemeData)
%
% Performs the actual work on a single level set function.
%
% Parameters (same as postTimestepReinit, except no vector level sets):
%   t              Current time.
%   yIn            Input version of the level set function, in vector form.
%   schemeData     A structure.
%
%   yOut           Output version of the level set function, in vector form.
%
% See help comments of postTimestepReinit for expected contents of the
%   schemeData structure.

  %---------------------------------------------------------------------------
  if(isfield(schemeData, 'reinitPerform') && ~schemeData.reinitPerform)
    % We are supposed to skip reinitialization.
    yOut = yIn;
    return;
  end

  %---------------------------------------------------------------------------
  % Only one required parameter.
  checkStructureFields(schemeData, 'grid');
  grid = schemeData.grid;

  %---------------------------------------------------------------------------
  % Create defaults for the rest.
  if(isfield(schemeData, 'reinitAccuracy'))
    accuracy = schemeData.reinitAccuracy;
  else
    accuracy = 'medium';
  end

  if(isfield(schemeData, 'reinitSteps'))
    if(isinf(schemeData.reinitSteps))
      % Enough time to pass right across the grid.
      tMax = max(grid.max - grid.min);
    else
      % We might be told to take zero (or a negative number of) steps.
      if(schemeData.reinitSteps < 1)
        yOut = yIn;
        return;
      end
      % To take a certain number of steps, pass tMax < 0.
      tMax = -schemeData.reinitSteps;
    end
  else
    % Enough time to pass through five grid nodes.
    tMax = 5 * max(grid.dx);
  end

  if(isfield(schemeData, 'reinitErrorMax'))
    errorMax = schemeData.reinitErrorMax;
  else
    errorMax = 1e-3;
  end

  %---------------------------------------------------------------------------
  % Perform the reinitialization.
  yOut = signedDistanceIterative(grid, yIn, accuracy, tMax, errorMax);
