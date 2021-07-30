function C = cellMatrixAdd(A, B)
% cellMatrixAdd: elementwise addition of cell matrices.
%
%   C = cellMatrixAdd(A,B)
%
% Let A(x) be an m by n matrix whose value depends on the state x.  In the
% level set toolbox, we have chosen to represent this matrix as an m by n
% cell matrix; entry A{i,j} is an array (the size of the grid) providing
% the (i,j) entry of A for each node in the grid.
%
% For example, grid.xs is a grid.dim by 1 cell matrix that provides the
% state x at each node in the grid.
%
% This function adds two cell matrices together: C(x) = A(x) + B(x).
% If A and B are m by n cell matrices then C is an m by n cell matrix.
%
% If either of A or B is not a cell matrix, then C is a cell matrix the
% size of the other one.  This case is the equivalent to adding a state
% dependent scalar term.  For example C(x) = A(x) + B(x), where B and C are
% cell matrices, and A is a regular array the same size as each cell entry
% of B.
%
% Each cell entry in A and B (or A and/or B themselves, if they are not 
% cell matrices) must be either a scalar or an array of identical size.
% The scalar case corresponds to a state independent entry.
%
% For example, if M = rand(3,1) and grid.dim = 3, then
% cellMatrixAdd(num2cell(M), grid.xs) will return a cell vector of size 3
% by 1.  Entry i of this cell vector will contain an array of size
% grid.shape which is the result of adding entry i of M to component i of
% the state vector x of each node in the grid.
%
% Input parameters:
%
%   A: Cell matrix of size m by n.
%
%   B: Cell matrix of size m by n.
%
% Output parameters:
%
%   C: Cell matrix of size m by n.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/23/04
% Subversion tags for version control purposes.
% $Date: 2011-05-16 16:06:25 -0700 (Mon, 16 May 2011) $
% $Id: cellMatrixAdd.m 66 2011-05-16 23:06:25Z mitchell $

  if(iscell(A))
    if(iscell(B))
      % Full matrix/matrix addition.
      sizeA = size(A);
      sizeB = size(B);      
      
      if((length(sizeA) ~= 2) || (length(sizeB) ~= 2))
        error('A and B must be cell arrays of dimension 2.');
      end
      if(any(sizeA ~= sizeB))
        error('Dimensions of A and B must match.');
      end
  
      C = cell(sizeA);
      for i = 1 : sizeA(1)
        for j = 1 : sizeA(2)
          C{i,j} = A{i,j} + B{i,j};
        end
      end
      return;
    
    elseif(isnumeric(B))
      % B will multiply every entry of A.
      scalar = B;
      array = A;
      
    else
      error('Input B must be a numeric array or a cell matrix');
    end
  
  elseif(isnumeric(A))
    if(iscell(B))
      % A will multiply every entry of B.
      scalar = A;
      array = B;
      
    elseif(isnumeric(B))
      % Regular pointwise array addition (including scalar + scalar).
      C = A + B;
      return;
      
    else
      error('Input B must be a numeric array or a cell matrix');
    end
  
  else
    error('Input A must be a numeric array or a cell matrix');
  end

  
  % If we drop through to here, one of the inputs was not a cell matrix.
  sizeArray = size(array);
  
  if(length(sizeArray) ~= 2)
    error('Cell array must be of dimension 2');
  end
  
  C = cell(sizeArray);
  for i = 1 : sizeArray(1);
    for j = 1 : sizeArray(2);
      C{i,j} = scalar + array{i,j};
    end
  end
