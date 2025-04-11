__all__ = ["HJ_MAD"]
__copyright__ 	= "2025, Hamilton-Jacobi Analysis in Python"
__comment__     = "Adaptive gradient descent on the Moreau envelope of the viscous HJ value function."
__credits__  	= "Haoxiang You, Ian Abraham."
__license__ 	= "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Completed"

import sys 
import torch 
import logging 
import numpy as np 
import scipy.ndimage as sp_ndimage

logger = logging.getLogger(__name__)


class HJ_MAD:
    ''' 
        Hamilton-Jacobi Moreau Stochastic Gradient Descent  is used to solve nonconvex minimization
        problems via a zeroth-order sampling scheme.
        
        Inputs:
          1)  dynamics     = Class that contains the dynamics of the agents, hamiltonian function and other auxiliary variables.
          2)  delta        = coefficient of viscous term in the HJ equation
          3)  int_samples  = number of samples used to approximate expectation in heat equation solution
          4)  t_span       = time vector containig [initial time, minimum time allowed, maximum time]
          6)  tol          = stopping tolerance
          7)  psi          = parameter used to update tk
          8) beta          = exponential averaging term for gradient beta (beta multiplies history, 1-beta multiplies current grad)
          9) eta_vec       = vector containing [eta_minus, eta_plus], where eta_minus < 1 and eta_plus > 1 (part of time update)
          10) alpha        = step size. has to be in between (1-sqrt(eta_minus), 1+sqrt(eta_plus))
          11) fixed_time   = boolean for using adaptive time
          12) verbose      = boolean for printing

        Outputs:
          1) x_opt                    = optimal x_value approximation
          2) xk_hist                  = update history
          3) tk_hist                  = time history
          4) fk_hist                  = function value history
          6) rel_grad_vk_norm_hist    = relative grad norm history of Moreau envelope
    '''
    def __init__(self, dynamics, delta=0.1, int_samples=100, t_span = [0, 1],  
                 tol=5e-2, psi=0.3, beta=0.9, alpha=1.0, adapt_time=False, verbose=True):   
      self.delta            = delta
      self.g                = dynamics.get_values
      self.int_samples      = int_samples
      self.tol              = tol
      self.t_span           = t_span
      self.psi              = psi
      self.beta             = beta 
      self.alpha            = alpha 
      self.adapt_time       = adapt_time
      self.verbose          = verbose

      # samples tools
      self.states           = dynamics.state_space
      self.probs            = torch.ones(self.states.shape[0], dtype=torch.float) / self.states.shape[0]
      
      'HJI Hyperparameters.'
      '--------------------'
      eps = sys.float_info.epsilon
      self.t_steps = (t_span[-1] - t_span[0]) / 100
      self.small = 100 * eps 

      self.dim = self.states.shape[1] 
      # check that alpha is in right interval
      assert(alpha >= 1-np.sqrt(0.9))
      assert(alpha <= 1+np.sqrt(1.1))

      self.dynamics = dynamics 
    
    def noise_samples(self, x_sampled, delta, t):
        """
          Corrupt the samples drawn from the state space with some 
          Gaussian noise.  
        """
        var = delta * t
        x_sampled = x_sampled.numpy()
        noised = np.zeros_like(x_sampled)
        for idx in range(x_sampled.shape[1]):
            noised[:, idx] = sp_ndimage.gaussian_filter(x_sampled[:, idx], var)

        return torch.Tensor(noised + x_sampled) #, dtype=torch.float64)
    
    def compute_grad_vk(self, x, t, g, delta): #, eps=1e-12):
      ''' 
          Compute the gradient of the Moreau envelope.
      '''
      y =  self.noise_samples(x, delta, t)
      
      exp_term = torch.exp(-g(y)/delta)
      phi_delta       = torch.mean(exp_term)

      numerator = y.t()*exp_term 
      numerator = torch.mean(numerator.t(), dim=0)      
      grad_vk = (x.squeeze() -  numerator/(phi_delta + self.small)) 
      
      hamiltonian = self.dynamics.hamiltonian(grad_vk, x)
      hamterm = torch.minimum(torch.Tensor([0]), -1*hamiltonian)

      vk       = -delta * torch.log(phi_delta+self.small)

      hji_rcbrt = vk + hamterm

      return grad_vk, vk, hji_rcbrt

    def update_time(self, t_now, rel_grad_vk_norm):
      '''
        time step rule

        if ‖gk_plus‖≤ psi (‖gk‖+ eps):
          min (eta_plus t,T)
        else
          max (eta_minus t,t_min) otherwise

        OR:
        
        if rel grad norm too small, increase tk (with maximum T).
        else if rel grad norm is too "big", decrease tk with minimum (t_min)

        ToDo: Haoxiang, can you change to Armijio's rule.
      '''
      if rel_grad_vk_norm <= self.psi:
        # increase t when relative gradient norm is smaller than psi
        logger.debug("Increasing time")
        t_now += self.t_steps
      else:
        logger.debug("Decreasing time")
        # decrease otherwise t when relative gradient norm is smaller than psi
        t_now -= self.t_steps

      return t_now

    def run(self):
      xk_hist               = np.zeros((1, 3))  
      xk_error_hist         = [] 
      rel_grad_vk_norm_hist = [] 
      gk_hist               = [1e9] 
      tk_hist               = [] 
      counter               = 0
      hji_rcbrt_term_hist   = []
      val_func              = []

      t_now                 = self.t_span[0]

      sample_idces = self.probs.multinomial(self.int_samples, replacement=True)
      xk     = self.states[sample_idces, :]
      x_opt = xk

      first_moment, vk, hji_rcbrt_term   = self.compute_grad_vk(xk, t_now, self.g, self.delta)
      val_func.append(vk)
      rel_grad_vk_norm      = torch.norm(vk, p=2).item() #1.0

      fmt = '[{:3d}] |  t_now = {:2.4f} | gk = {:3.4f} | xk_err = {:3.4f} '
      fmt += ' | |grad_vk| = {:3.4f} | hj_term = {:2.2f} '

      print('\n')
      logger.info('-------------------------- RUNNING HJ-MAD ---------------------------')
      logger.info(f'dimension = f{self.dim}, n_samples = {self.int_samples}')

      converged = True
      while converged and (self.t_span[1]  - t_now) > self.small * self.t_span[1]:  
        
        xk_norm = torch.norm(xk, p=2, dim=0)

        func_eval = self.g(xk)
        
        counter += 1
        t_now = t_now + self.t_steps

        rel_grad_vk_norm = rel_grad_vk_norm.item() if isinstance(rel_grad_vk_norm, torch.Tensor) else rel_grad_vk_norm

        xk_hist = np.vstack((xk_hist, xk_norm.numpy()))
        rel_grad_vk_norm_hist.append(rel_grad_vk_norm)
        xk_error_hist.append(np.linalg.norm(xk_hist[-1]-xk_hist[-2], ord=2).tolist())
        tk_hist.append(t_now)
        gk_hist.append(torch.norm(torch.norm(func_eval,  2, dim=0), p=2).item())
        hji_rcbrt_term_hist.append(torch.norm(hji_rcbrt_term, p=2, dim=0).item())

        if self.verbose:
          print(fmt.format(counter, t_now, gk_hist[-1], np.linalg.norm(xk_hist[-1]).item(), rel_grad_vk_norm_hist[-1], hji_rcbrt_term_hist[-1]))

        if np.all(np.all(np.abs(xk_error_hist[:-3]))<self.tol):
          tk_hist = tk_hist[1:]
          xk_hist = xk_hist[1:]
          xk_error_hist = xk_error_hist[1:]
          rel_grad_vk_norm_hist = rel_grad_vk_norm_hist[1:]
          gk_hist               = gk_hist[1:]
          logger.info('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_vk_norm_hist[-1]))
          logger.info(f'iter = ', t_now, ', number of function evaluations: {len(xk_error_hist)*self.int_samples}')
          converged = False  

        if counter>10 and (gk_hist[-1] < gk_hist[-2]) and (gk_hist[-2] < gk_hist[-3]):
          x_opt = xk 

        xk -= self.alpha * first_moment 
        
        grad_vk, vk, hji_rcbrt_term = self.compute_grad_vk(xk, t_now, self.g, self.delta)
        val_func.append(vk)

        if  self.adapt_time:
          t_now = self.update_time(t_now, rel_grad_vk_norm)

        grad_vk_norm_old = torch.norm(first_moment)
        first_moment  = self.beta*first_moment + (1-self.beta)*grad_vk

        grad_vk_norm = torch.norm(first_moment)
        rel_grad_vk_norm = grad_vk_norm/(grad_vk_norm_old + 1e-12)

      return x_opt, xk_hist, xk_error_hist, tk_hist, rel_grad_vk_norm_hist, gk_hist, val_func

