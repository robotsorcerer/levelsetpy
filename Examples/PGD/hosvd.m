function [Core, Un,  V, ranks] = hosvd(X, eps, plot)
% Using a user defined projection error as a criteria,
% calculate the minimum projection error onto a subspace of the 
% value function X. 
%
%   Parameters:
%   -----------
%       X(matrix of doubles): Payoff functional; 
%       V (matrix of doubles): (Dominant) Left singular vectors;
%       eps (float): (User-defined) projection error;
%       plot(bool):  show the projection error on a plot?
%
%   Returns:
%   --------
%   Core: Core tensor (representing the critical mass of X)
%   Un: Optimal orthonormal matrix (see paper)
%   V: Optimal eigen vectors
%   ranks: best rank approximation of X that admits the tolerable decomp
%            error
%       Basically, it is a projection of X onto one of its subspaces.
%   
% Author: Lekan Molu, Dec 3, 2021
    
    % Get gram matrix
    S = X * X';
    
    % do eigen decomposition of Gram matrix
    [V, Lambda] = eig(S);
    
    %Find Un
    Sig = diag(Lambda, 0);  % collect 0-th diagonal elements
    err_term = (eps^2 * norm(X, 'fro')^2)/ndims(X) ;
    
    lambs = cumsum(Sig(2:end));
    ranks = nnz(lambs <= err_term);     
    Un = V(:, 1:ranks);
    
    Core = X * Un;
end
  