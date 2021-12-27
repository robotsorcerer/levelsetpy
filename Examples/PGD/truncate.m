function [Vc, U] = truncate(V, eps)
%{
    Truncate a high-order tensor to a best rank-1, rank-2, ...
    tensor using alg 3 in the paper.
    
    Parameters
    ----------
    V: Value Function
    eps: truncation accuracy
%}