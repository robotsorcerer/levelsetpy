function traceA = cellMatrixTrace(A)
% cellMatrixTrace: (state dependent) trace of a matrix A(x)
%
%   traceA = cellMatrixTrace(A)
%
% For a spatially varying matrix A stored as a cell matrix (ie A{i,j} is an
% array which gives the value of A(i,j) at each node) this function returns
% a regular array whose entries are the trace of A at each node.
%
% If A is a regular matrix, it is treated as a spatially varying scalar
% (and trace(scalar) = scalar).
%
% Input parameters:
%
%   A: Spatially varying matrix stored as a cell matrix.
%
% Output parameters:
%
%   traceA: Regular array containing the trace, so that
%   traceA(i,j) = sum_k A{k,k}(i,j)

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/20/04
% Subversion tags for version control purposes.
% $Date: 2011-05-14 23:23:02 -0700 (Sat, 14 May 2011) $
% $Id: cellMatrixTrace.m 63 2011-05-15 06:23:02Z mitchell $

if(ndims(A) ~= 2)
  error('cellMatrixMax only works on 2D cell arrays');
end

if(iscell(A))
  traceA = 0;
  for k = 1 : min(size(A))
    traceA = traceA + A{k,k};
  end
else
  traceA = A;
end
