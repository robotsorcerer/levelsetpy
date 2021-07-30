function [ ydot, stepBound, schemeData ] = termTraceHessian(t, y, schemeData)
% termTraceHessian: approximate update by the trace of the Hessian
%
% [ ydot, stepBound, schemeData ] = termTraceHessian(t, y, schemeData)
%
% Computes an approximation to
%
%		- trace(L(x,t) D_x^2 \phi R(x,t))
%
%   where L(x,t) and R(x,t) are matrices (may be state dependent).  This second
%   order equation can be used for Folker-Planck or Kolmogorov PDEs
%   for computing expected values of SDEs when the underlying SDE
%   has an input-free stochastic term.  Choose L = R = eye(grid.dim)
%   to get an approximation of the Laplacian (rather slow).
%
% Derived from methods designed for curvature dependent flow 
%   in O&F, chapters 4.1 & 4.2.
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
%   .hessianFunc   Function handle to finite difference approximation of
%                    the Hessian (matrix of second order derivatives).
%   .L		   Matrix by which to left multiply the Hessian (see below).
%   .R		   Matrix by which to right multiply the Hessian (see below).
%
% It may contain additional fields at the user's discretion.
%
% schemeData.L and schemeData.R may provide matrices in one of three ways
%
%   1) For time and space invariant matrices, an n x n matrix.  The same
%      matrix will be used on the Hessian for every node of the grid.
%
%   2) For time invariant but spatially dependent matrices, an n x n
%      cell matrix, each element of which is an array the size of data.
%      The function cellMatrixMultiply is used to apply such matrices to
%      the Hessian.
%
%   3) For general matrices, a function handle to a function with prototype
%                  M = matrixGridFunc(t, data, schemeData)
%      where output M is a matrix or cell matrix from (1) or (2) and the
%      input arguments are the same as those of this function (except that
%      data = y has been reshaped to its original size).  In this case
%      it may be useful to include additional fields in schemeData.
%
%  where n is the dimension of the grid.
%
% For evolving vector level sets, y may be a cell vector.  If y is a cell
%   vector, schemeData may be a cell vector of equal length.  In this case
%   all the elements of y (and schemeData if necessary) are ignored except
%   the first.  As a consequence, if schemeData.L and/or schemeData.R 
%   is a function handle the call to matrixGridFunc will be performed with 
%   a regular data array and a single schemeData structure 
%   (as if no vector level set was present).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell 8/20/04
% Updated to handle vector level sets.  Ian Mitchell 11/23/04.

  %---------------------------------------------------------------------------
  % For vector level sets, ignore all the other elements.
  if(iscell(schemeData))
    thisSchemeData = schemeData{1};
  else
    thisSchemeData = schemeData;
  end

  checkStructureFields(thisSchemeData, 'grid', 'hessianFunc', 'L', 'R');

  grid = thisSchemeData.grid;

  %---------------------------------------------------------------------------
  if(iscell(y))
    data = reshape(y{1}, grid.shape);    
  else
    data = reshape(y, grid.shape);
  end

  %---------------------------------------------------------------------------
  % Get matrices.
  L = getMatrix(t, data, thisSchemeData, thisSchemeData.L);
  R = getMatrix(t, data, thisSchemeData, thisSchemeData.R);

  %---------------------------------------------------------------------------
  % Get Hessian (which will be a symmetric cell matrix with no upper right).
  P = feval(thisSchemeData.hessianFunc, grid, data);

  % Fill in the upper right
  for i = 1 : grid.dim
    for j = i+1 : grid.dim
      P{i,j} = P{j,i};
    end
  end

  %---------------------------------------------------------------------------
  % Compute trace(L * P * R)
  A = cellMatrixMultiply(cellMatrixMultiply(L, P), R);
  update = cellMatrixTrace(A);

  %---------------------------------------------------------------------------
  % Compute CFL timestep bound (formula is currently a guess).
  D = num2cell(1 ./ (grid.dx * grid.dx'));
  timeStepMatrix = cellMatrixMultiply(cellMatrixMultiply(L, D), R);
  timeStepTrace = cellMatrixTrace(timeStepMatrix);
  stepBound = 1 / (2 * max(abs(timeStepTrace(:))));

  %---------------------------------------------------------------------------
  % Reshape output into vector format.
  %   We do not need to negate for RHS of ODE: since we skipped the
  %   negation in the trace step above.
  ydot = update(:);



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function outM = getMatrix(t, data, schemeData, inM)
% getMatrix: transform the three possible matrix forms into a cell matrix.
%
%   outM = getMatrix(t, data, schemeData, inM)
%
% The input inM may be in one of three formats
%   (see help entry for termTraceHessian for more details)
%
%     1) a scalar matrix.
%     2) a cell matrix.
%     3) a function handle to a function which returns (1) or (2).
%
% This function converts all of these possibilities into a matrix of type (2).
%
% parameters:
%   t            Time at beginning of timestep.
%   data         Data array.
%   schemeData	 A structure (see above).
%   inM          Input matrix of type (1), (2) or (3).
%
%   outM         Output matrix of type (2).

  %---------------------------------------------------------------------------
  % If the input is a function handle, call the function.
  if(isa(inM, 'function_handle'))
    outM = feval(inM, t, data, schemeData);
  else
    outM = inM;
  end

  %---------------------------------------------------------------------------
  % Convert result to cell matrix, if necessary
  if(isa(outM, 'double'))
    outM = num2cell(outM);
  elseif(~isa(outM, 'cell'))
    error('Input matrix must be a matrix, cell matrix, or function handle.');
  end
