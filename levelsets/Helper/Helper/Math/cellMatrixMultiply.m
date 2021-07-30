function C = cellMatrixMultiply(A, B)
% cellMatrixMultiply: elementwise multiplication of cell matrices
%
%   C = cellMatrixMultiply(A,B)
%
% Let A(x) be an m by n matrix whose value depends on the state x.  In the
% level set toolbox, we have chosen to represent this matrix as an m by n
% cell matrix; entry A{i,j} is an array (the size of the grid) providing
% the (i,j) entry of A for each node in the grid.
%
% For example, grid.xs is a grid.dim by 1 cell matrix that provides the
% state x at each node in the grid.
%
% This function multiplies two cell matrices together: C(x) = A(x) * B(x).
% If A is an m by n cell matrix and B is an n by p cell matrix, then C is
% an m by p cell matrix.  
%
% If either of A or B is not a cell matrix, then C is a cell matrix the size
% of the other one.  This case is the equivalent of multiplying by a 
% state dependent scalar term.  For example C(x) = A(x) * B(x), where
% B and C are cell matrices, and A is a regular array the same size as 
% each cell entry of B.
%
% Each cell entry in A and B (or A and/or B themselves, if they are not 
% cell matrices) must be either a scalar or an array of identical size.
% The scalar case corresponds to a state independent entry.
%
% For example, if M = rand(3,3) and grid.dim = 3, then
% cellMatrixMultiply(num2cell(M), grid.xs) will return a cell vector of
% size 3 by 1.  Entry i of this cell vector will contain an array of size
% grid.shape which is the result of multiplying row i of M by the state
% vector x of each node in the grid.
%
% Note that if A or B is a pure scalar, this function is equivalent to
% multiplying every element of every cell entry of the other by that
% scalar.
%
% Input parameters:
%
%   A: Cell matrix of size m by n.
%
%   B: Cell matrix of size n by p.
%
% Output parameters:
%
%   C: Cell matrix of size m by p.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 2/19/04
% Subversion tags for version control purposes.
% $Date: 2011-05-14 23:23:02 -0700 (Sat, 14 May 2011) $
% $Id: cellMatrixMultiply.m 63 2011-05-15 06:23:02Z mitchell $

  if(iscell(A))
    if(iscell(B))
      % Full matrix/matrix multiplication
      sizeA = size(A);
      sizeB = size(B);      
      
      if((length(sizeA) ~= 2) || (length(sizeB) ~= 2))
        error('A and B must be cell arrays of dimension 2.');
      end
      if(sizeA(2) ~= sizeB(1))
        error('Inner dimensions of A and B must match.');
      end
  
      C = cell(sizeA(1), sizeB(2));
      for i = 1 : sizeA(1)
        for j = 1 : sizeB(2)
          C{i,j} = A{i,1} .* B{1,j};
          for k = 2 : sizeA(2)
            C{i,j} = C{i,j} + A{i,k} .* B{k,j};
          end
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
      % Regular pointwise array multiplication (including scalar * scalar).
      C = A .* B;
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
      C{i,j} = scalar .* array{i,j};
    end
  end
