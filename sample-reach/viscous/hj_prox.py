import numpy as np
import torch

def compute_prox(x, t, f, delta=1e-1, int_samples=100, alpha=1.0, linesearch_iters=0, device='cpu'):
    """ Estimate proximals from function value sampling via HJ-Prox Algorithm.

        The output estimates the proximal:
        
        $$
            \mathsf{prox_{tf}(x) = argmin_y \ f(y) + \dfrac{1}{2t} \| y - x \|^2,}
        $$
            
        where $\mathsf{x}$ = `x` is the input, $\mathsf{t}$=`t` is the time parameter, 
        and $\mathsf{f}$=`f` is the function of interest. The process for this is 
        as follows.
        
        - [x] Sample points $\mathsf{y^i}$ (via a Gaussian) about the input $\mathsf{x}$
        - [x] Evaluate function $\mathsf{f}$ at each point $\mathsf{y^i}$
        - [x] Estimate proximal by using softmax to combine the values for $\mathsf{f(y^i)}$ and $\mathsf{y^i}$            

        Note: 
            The computation for the proximal involves the exponential of a potentially
            large negative number, which can result in underflow in floating point
            arithmetic that renders a grossly inaccurate proximal calculation. To avoid
            this, the "large negative number" is reduced in size by using a smaller
            value of alpha, returning a result once the underflow is not considered
            significant (as defined by the tolerances "tol" and "tol_underflow").
            Utilizing a scaling trick with proximals, this is mitigated by using
            recursive function calls.
            
        Warning:
            Memory errors can occur if too many layers of recursion are used,
            which can happen with tiny delta and large f(x). 

        Args:
            x (tensor): Input vector
            t (tensor): Time > 0
            f (Callable): Function to minimize
            delta (float, optional): Smoothing parameter
            int_samples (int, optional): Number of samples in Monte Carlo sampling for integral
            alpha (float, optional): Scaling parameter for sampling variance
            linesearch_iters (int, optional): Number of steps used in recursion (used for numerical stability)
            device (string, optional): Device on which to store variables

        Shape:
            - Input `x` is of size `(n, 1)` where `n` is the dimension of the space of interest
            - The output `prox_term` also has size `(n, 1)`

        Returns:
            prox_term (tensor): Estimate of the proximal of f at x
            linesearch_iters (int): Number of steps used in recursion (used for numerical stability)
            envelope (tensor): Value of envelope function (i.e. infimal convolution) at proximal
            
        Example:
            Below is an exmaple for estimating the proximal of the L1 norm. Note the function
            must have inputs of size `(n_samples, n)`.
            ```
                def f(x):
                    return torch.norm(x, dim=1, p=1) 
                n = 3
                x = torch.randn(n, 1)
                t = 0.1
                prox_term, _, _ = compute_prox(x, t, f, delta=1e-1, int_samples=100)   
            ```
    """
    assert x.shape[1] == 1
    assert x.shape[0] >= 1
    
    linesearch_iters +=1
    standard_dev = np.sqrt(delta * t / alpha)
    dim = x.shape[0]
    
    y = standard_dev * torch.randn(int_samples, dim, device=device) + x.permute(1,0) # y has shape (n_samples, dim)
    z = -f(y)*(alpha/delta)     # shape =  n_samples
    w = torch.softmax(z, dim=0) # shape = n_samples 
    
    softmax_overflow = 1.0 - (w < np.inf).prod()
    if softmax_overflow:
        alpha *= 0.5
        return compute_prox(x, t, f, delta=delta, int_samples=int_samples, alpha=alpha,
                            linesearch_iters=linesearch_iters, device=device)
    else:
        prox_term = torch.matmul(w.t(), y)
        prox_term = prox_term.view(-1,1)
    
    prox_overflow = 1.0 - (prox_term < np.inf).prod()
    assert not prox_overflow, "Prox Overflowed"

    envelope = f(prox_term.view(1,-1)) + (1/(2*t)) * torch.norm(prox_term - x.permute(1,0), p=2)**2    
    return prox_term, linesearch_iters, envelope