__all__ = ["HJ_MAD"]

# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# Adapted from https://github.com/mines-opt-ml/hj-global-opt.git
# ------------------------------------------------------------------------------------------------------------
import torch 
import numpy as np 


# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# ------------------------------------------------------------------------------------------------------------


class HJ_MAD:
    ''' 
        Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) is used to solve nonconvex minimization
        problems via a zeroth-order sampling scheme.
        
        Inputs:
          1)  g           = function to be minimized. Inputs have size (n_samples x n_features). Outputs have size n_samples
          3)  delta        = coefficient of viscous term in the HJ equation
          4)  int_samples  = number of samples used to approximate expectation in heat equation solution
          6)  t_vec        = time vector containig [initial time, minimum time allowed, maximum time]
          7)  max_iters    = max number of iterations
          8)  tol          = stopping tolerance
          9)  psi          = parameter used to update tk
          10) beta         = exponential averaging term for gradient beta (beta multiplies history, 1-beta multiplies current grad)
          11) eta_vec      = vector containing [eta_minus, eta_plus], where eta_minus < 1 and eta_plus > 1 (part of time update)
          11) alpha        = step size. has to be in between (1-sqrt(eta_minus), 1+sqrt(eta_plus))
          12) fixed_time   = boolean for using adaptive time
          13) verbose      = boolean for printing

        Outputs:
          1) x_opt                    = optimal x_value approximation
          2) xk_hist                  = update history
          3) tk_hist                  = time history
          4) fk_hist                  = function value history
          6) rel_grad_uk_norm_hist    = relative grad norm history of Moreau envelope
    '''
    def __init__(self, g, x_true, delta=0.1, int_samples=100, t_vec = [1.0, 1e-3, 1e1], max_iters=5e4, 
                 tol=5e-2, psi=0.9, beta=[0.9], eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, verbose=True):
      
      self.delta            = delta
      self.g                = g
      self.int_samples      = int_samples
      self.max_iters        = max_iters
      self.tol              = tol
      self.t_vec            = t_vec
      self.psi              = psi
      self.x_true           = x_true
      self.beta             = beta 
      self.alpha            = alpha 
      self.eta_vec          = eta_vec
      self.fixed_time       = fixed_time
      self.verbose          = verbose
      
      self.dim = 3; 
      # check that alpha is in right interval
      assert(alpha >= 1-np.sqrt(eta_vec[0]))
      assert(alpha <= 1+np.sqrt(eta_vec[1]))
    
    def compute_grad_uk(self, x, t, g, delta, eps=1e-12):
      ''' 
          Compute the gradient og the Moreau envelope.
      '''

      standard_dev = np.sqrt(delta*t)

      n_features = x.shape[0]
      y = standard_dev * torch.randn(self.int_samples, n_features) + x
      
      exp_term = torch.exp(-g(y)/delta)
      phi_delta       = torch.mean(exp_term)

      # separate grad_v into two terms for numerical stability
      numerator = y*exp_term.view(self.int_samples, 1)
      numerator = torch.mean(numerator, dim=0)
      grad_uk = (x -  numerator/(phi_delta + eps)) # the t gets canceled with the update formula

      uk       = -delta * torch.log(phi_delta+eps)

      return grad_uk, uk

    def update_time(self, tk, rel_grad_uk_norm):
      '''
        time step rule

        if ‖gk_plus‖≤ psi (‖gk‖+ eps):
          min (eta_plus t,T)
        else
          max (eta_minus t,t_min) otherwise

        OR:
        
        if rel grad norm too small, increase tk (with maximum T).
        else if rel grad norm is too "big", decrease tk with minimum (t_min)
      '''

      eta_minus = self.eta_vec[0]
      eta_plus = self.eta_vec[1]
      T = self.t_vec[2]
      t_min = self.t_vec[1]

      if rel_grad_uk_norm <= self.psi:
        # increase t when relative gradient norm is smaller than psi
        tk = min(eta_plus*tk , T) 
      else:
        # decrease otherwise t when relative gradient norm is smaller than psi
        tk = max(eta_minus*tk, t_min)

      return tk

    def run(self, x0):

      n_features            = x0.shape[0]
      print(f'max_iters: {type(self.max_iters)}, n_features: {n_features}, {type(n_features)}')
      xk_hist               = torch.zeros(int(self.max_iters), n_features)
      xk_error_hist         = torch.zeros(self.max_iters)
      rel_grad_uk_norm_hist = torch.zeros(self.max_iters)
      gk_hist               = torch.zeros(self.max_iters)
      tk_hist               = torch.zeros(self.max_iters)
      counter               = 1

      xk    = x0
      x_opt = xk
      tk    = self.t_vec[0]
      t_max = self.t_vec[2]

      first_moment, _       = self.compute_grad_uk(xk, tk, self.g, self.delta)
      rel_grad_uk_norm      = 1.0

      fmt = '[{:3d}]: gk = {:6.2e} | xk_err = {:6.2e} '
      fmt += ' | |grad_uk| = {:6.2e} | tk = {:6.2e}'

      print('-------------------------- RUNNING HJ-MAD ---------------------------')
      print('dimension = ', self.dim, 'n_samples = ', self.int_samples)

      for k in range(self.max_iters):

        xk_hist[k,:]    = xk

        rel_grad_uk_norm_hist[k]  = rel_grad_uk_norm

        xk_error_hist[k] = torch.norm(xk - self.x_true)
        tk_hist[k]       = tk

        # gk_hist[k]       = self.g(xk.view(1, n_features))
        gk_hist[k]       = self.g(xk) #.view(1, n_features))

        if self.verbose:
          print(fmt.format(k+1, gk_hist[k], rel_grad_uk_norm_hist[k], tk))

        if xk_error_hist[k] < self.tol:
          tk_hist = tk_hist[0:k+1]
          xk_hist = xk_hist[0:k+1,:]
          xk_error_hist = xk_error_hist[0:k+1]
          rel_grad_uk_norm_hist = rel_grad_uk_norm_hist[0:k+1]
          gk_hist               = gk_hist[0:k+1]
          print('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
          print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          break
        elif k==self.max_iters-1:
          print('HJ-MAD failed to converge with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
          print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          print('Used fixed time = ', self.fixed_time)

        if k>0:
          if gk_hist[k] < gk_hist[k-1]:
            x_opt = xk 

        xk = xk - self.alpha * first_moment # tk gets canceled out with gradient formula
        
        if self.fixed_time == False:
          tk = self.update_time(tk, rel_grad_uk_norm)
        
        grad_uk, _ = self.compute_grad_uk(xk, tk, self.f, self.delta)

        grad_uk_norm_old = torch.norm(first_moment)
        first_moment  = self.beta*first_moment + (1-self.beta)*grad_uk

        grad_uk_norm = torch.norm(first_moment)
        rel_grad_uk_norm = grad_uk_norm/(grad_uk_norm_old + 1e-12)

      return x_opt, tk_hist, rel_grad_uk_norm_hist, gk_hist