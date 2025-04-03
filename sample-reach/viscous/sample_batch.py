#!/usr/bin/env python
# coding: utf-8

# In[47]:


import torch 
from torch.utils.data import Dataset 

import numpy as np

import sys, os
from gmm import GMM
import scipy.ndimage as sp_ndimage
from rockets import RocketDynamics

import matplotlib 
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# ## Problem Statement and Numerical Setting

# Let $\Omega = [0,L]\subset \mathbb{R}$ be the spatial domain indicated by the variable $\omega$, and let $(0,T]\subset\mathbb{R}$ be the time domain with variable $t$.
# We consider the three-dimensional HJ equation with homogeneous Dirichlet boundary conditions,
# $$
# \begin{align*}
#     &\partial \bm{v}_t(t; \bm{x}) + \min\{0, \bm{H}\left(x, \partial \bm{v}_{\bm{x}}(t; \bm{x})\right)\} = 0, \qquad &\bm{v}(0; \bm{x}) = \bm{g}(0; \bm{x}) \nonumber \\
# %\bm{v}(0; \bm{x}) &= \bm{g}(0; \bm{x}) \\
# %
# &\approx \partial \bm{v}^\delta_t(t; \bm{x}) + \min\{0, 
# \bm{H}^\delta\left(t; x, \partial \bm{v}^\delta_{\bm{x}}\right)\} = 0, \qquad &\bm{v}^\delta(0; \bm{x}) = \bm{g}(0; \bm{x}) \nonumber \\
# %
# &:= \partial \bm{v}^\delta_t(t; \bm{x}) + \min\bigg\{0, 
# \max_{\bm{u} \in \mathcal{U}} \min_{\bm{w} \in \mathcal{W}} \, \bigg\langle f(t; \bm{x}, \bm{u}, \bm{w}), \frac{1}{t}(\bm{x} - \text{prox}_{t\bm{g}}(\bm{x})) \bigg\rangle
# \bigg\} = 0 \qquad &\bm{v}^\delta(0; \bm{x}) = \bm{g}(0; \bm{x}).
# \end{align*}
# $$
# 
# This is a model for a one-dimensional rod that conducts heat: the temperature at the ends of the rod are fixed at $0$ and heat is allowed to flow out of the rod through the ends.

# In[ ]:


def termRestrictUpdate(t, values, LaxFriedrichs):
    """
     termRestrictUpdate: restrict the sign of a term to be positive or negative.

     [ ydot, stepBound, LaxFriedrichs ] = termRestrictUpdate(t, y, LaxFriedrichs)

     Given some other HJ term approximation, this function either restricts
       the sign of that update to be either:
           nonnegative (level set function only decreases), or
           nonpositive (level set function only increases).

     The CFL restriction of the other HJ term approximation is returned
       without modification.

     Parameters:
     ----------
       t            Time at beginning of timestep.
       y            Data array in vector form.
       LaxFriedrichs	 LaxFriedrichs Integration Scheme.
     
     Returns:
     -------
       ydot	 Change in the data array, in vector form.
       stepBound	 CFL bound on timestep for stability.
       LaxFriedrichs structure that leverages the proximal operator's costates #(see below).

     LaxFriedrichs is a structure containing data specific to this type of
       term approximation.  For this function it contains the field(s)

       .innerFunc   Function handle for the approximation scheme to
                      calculate the unrestricted HJ term.
       .innerData   LaxFriedrichs structure to pass to innerFunc.
       .positive    Boolean, true if update must be positive (nonnegative)
                      (optional, default = 1).

     It may contain addition fields at the user's discretion.

     While termRestrictUpdate will not change the LaxFriedrichs structure,
       the LaxFriedrichs.innerData component may be changed during the call
       to the LaxFriedrichs.innerFunc function handle.

     For evolving vector level sets, y may be a cell vector.  In this case
       the entire y cell vector is passed unchanged in the call to the
       LaxFriedrichs.innerFunc function handle.

     If y is a cell vector, LaxFriedrichs may be a cell vector of equal length.
       In this case, LaxFriedrichs[0] contains the fields listed above.  In the
       call to LaxFriedrichs[0].innerFunc, the LaxFriedrichs cell vector is passed
       unchanged except that the element LaxFriedrichs[0] is replaced with
       the contents of the LaxFriedrichs[0].innerData field.

      Lekan Molu, 08/21/21
    """
    ham_compute = LaxFriedrichs.hamFunc(t, values, LaxFriedrichs.CoStates)

    #Get the unrestricted update. # this is usually termLaxFriedrichs
    unRestricted, stepBound, innerData = termLaxFriedrichs(t, y, CoStates)

    ydot = torch.maximum(unRestricted, 0).squeeze()

    return ydot, stepBound, LaxFriedrichs


# In[ ]:


class HJ_MAD:
    ''' 
        Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) is used to solve nonconvex minimization
        problems via a zeroth-order sampling scheme.
        
        Inputs:
          1)  g           = function to be minimized. Inputs have size (n_samples x n_features). Outputs have size n_samples
          2)  x_true       = true global minimizer [N X D] sized where N is total number of discretized points on the state space and D is the dim of the states.
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
          6) rel_grad_vk_norm_hist    = relative grad norm history of Moreau envelope
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

      'HJ hyperparams'
      eps = sys.float_info.epsilon
      self.t_steps = (t_vec[-1] - t_vec[0]) / 10
      self.small = 100 * eps 

      self.dim = x_true.shape[1] 
      # check that alpha is in right interval
      assert(alpha >= 1-np.sqrt(eta_vec[0]))
      assert(alpha <= 1+np.sqrt(eta_vec[1]))
    
    def noise_samples(self, x0, delta, t):
        # kernel.shape 
        var = delta * t
        x0 = x0.numpy()
        noised = np.zeros_like(x0)
        for idx in range(x0.shape[1]):
            noised[:, idx] = sp_ndimage.gaussian_filter(x0[:, idx], var)

        return torch.Tensor(noised + x0)
    
    def compute_grad_vk(self, x, t, g, delta): #, eps=1e-12):
      ''' 
          Compute the gradient of the Moreau envelope.
      '''

      standard_dev = np.sqrt(delta*t)

      n_features = x.shape[1]
      # print(f'self.int_samples: {self.int_samples} n_features {n_features} x.shape: {x.shape}')
      # y = standard_dev * torch.randn(self.int_samples, n_features) + x #.view(1, -1)
      # y = standard_dev * torch.randn(x.shape) + x #.view(1, -1)
      y = self.noise_samples(x, delta, t)
      
      exp_term = torch.exp(-g(y)/delta)
      phi_delta       = torch.mean(exp_term)

      # separate grad_v into two terms for numerical stability
      # print(f'y: {y.shape} exp_term: {exp_term.shape}')
      numerator = y.t()*exp_term 
      numerator = torch.mean(numerator.t(), dim=0)
      # print(f'x: {x.shape} | numerator: {numerator.shape}')
      grad_vk = (x.squeeze() -  numerator/(phi_delta + self.small)) #.view(-1, 1) # the t gets canceled with the update formula

      vk       = -delta * torch.log(phi_delta+self.small)

      return grad_vk, vk

    def update_time(self, tk, rel_grad_vk_norm):
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
      T = self.t_vec[-1]
      t_min = self.t_vec[0]

      if rel_grad_vk_norm <= self.psi:
        # increase t when relative gradient norm is smaller than psi
        tk = min(eta_plus*tk , T) 
      else:
        # decrease otherwise t when relative gradient norm is smaller than psi
        tk = max(eta_minus*tk, t_min)

      return tk

    def run(self, x0):

      n_features            = x0.shape[0]
      xk_hist               = []  #torch.zeros(int(self.max_iters), n_features)
      xk_error_hist         = [] #torch.zeros(self.max_iters)
      rel_grad_vk_norm_hist = [] #torch.zeros(self.max_iters)
      gk_hist               = [] #torch.zeros(self.max_iters)
      tk_hist               = [] #torch.zeros(self.max_iters)
      counter               = 1

      xk    = x0
      x_opt = xk
      t_now    = self.t_vec[0]
      # t_max = self.t_vec[-1]

      first_moment, _       = self.compute_grad_vk(xk, t_now, self.g, self.delta)
      rel_grad_vk_norm      = 1.0

      fmt = '[{:3.4f}]: gk = {:6.2e} | xk_err = {:6.2e} '
      fmt += ' | |grad_vk| = {:6.2e} | tk = {:6.2e}'

      print('-------------------------- RUNNING HJ-MAD ---------------------------')
      print('dimension = ', self.dim, 'n_samples = ', self.int_samples)

      # for k in range(self.max_iters):
      t_now = self.t_vec[0]
      while (self.t_vec[1]  - t_now) > self.small * self.t_vec[1]:
        time_step = f"{t_now:.2f}/{self.t_vec[-1]}"

        rel_grad_vk_norm_hist.append(rel_grad_vk_norm)

        xk_error_hist.append(torch.norm((xk - self.x_true.squeeze())))
        tk_hist.append(t_now)

        gk_hist.append(torch.linalg.norm(self.g(xk.squeeze()), 2))

        if self.verbose:
          print(fmt.format(t_now, gk_hist[-1], xk_error_hist[-1], rel_grad_vk_norm_hist[-1], t_now))

			  # How far to step?
        t_vec = np.hstack([ t_now, min(self.t_vec[1], t_now + self.t_steps) ])

        if xk_error_hist[-1] < self.tol:
          tk_hist = tk_hist[0:-1]
          xk_hist = xk_hist[0:-1,:]
          xk_error_hist = xk_error_hist[0:-1]
          rel_grad_vk_norm_hist = rel_grad_vk_norm_hist[0:-1]
          gk_hist               = gk_hist[0:-1]
          print('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_vk_norm_hist[-1]))
          print('iter = ', t_now, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          break
        # elif k==self.max_iters-1:
        #   print('HJ-MAD failed to converge with rel grad norm {:6.2e}'.format(rel_grad_vk_norm_hist[k]))
        #   print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
        #   print('Used fixed time = ', self.fixed_time)

        if t_now>0:
          if gk_hist[-1] < gk_hist[-2]:
            x_opt = xk 

        print(f'xk: {xk.shape} | first_moment: {first_moment.shape}')
        xk -= self.alpha * first_moment # tk gets canceled out with gradient formula
        
        grad_vk, _ = self.compute_grad_vk(xk, t_now, self.g, self.delta)

        if self.fixed_time == False:
          t_now = self.update_time(t_now, rel_grad_vk_norm)

        grad_vk_norm_old = torch.norm(first_moment)
        first_moment  = self.beta*first_moment + (1-self.beta)*grad_vk

        grad_vk_norm = torch.norm(first_moment)
        rel_grad_vk_norm = grad_vk_norm/(grad_vk_norm_old + 1e-12)

      return x_opt, xk_hist, xk_error_hist, tk_hist, rel_grad_vk_norm_hist, gk_hist


# In[75]:


resolution = 1000
dynamics = RocketDynamics(1, 1, T=1, L=100, a=1, g=32, resolution=resolution)

torch.manual_seed(123)
np.random.seed(123)

x_all = dynamics.state_space
x0 = dynamics.get_initial_conditions()

target_region_size = resolution//4
'set target region as top quadrant of the state space'
x_true = torch.zeros_like(x0)
x_true[:target_region_size, 0] = x_all[:target_region_size, 0]
x_true[:target_region_size, 1] = x_all[:target_region_size, 1]
x_true[:target_region_size, 2] = x_all[:target_region_size, 2]

int_samples     = 100
max_iters       = int(5e4)

print(x0.shape, x_all.shape)


# In[ ]:


hj_mad_algo = HJ_MAD(dynamics.get_values, x_true, delta=0.1, int_samples=int_samples, t_vec = [0, 1.0], max_iters=int(5e4), 
                tol=5e-2, psi=0.9, beta=0.9, eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, verbose=True)

# run 30 times 
avg_trials      = 30
avg_func_evals  = 0



for i in range(avg_trials):
  x_opt_MAD, xk_hist_MAD, tk_hist_MAD, xk_error_hist_MAD, \
    rel_grad_uk_norm_hist_MAD, globalk_hist_MAD = hj_mad_algo.run(x0)
  avg_func_evals = avg_func_evals + len(xk_error_hist_MAD)*int_samples

avg_func_evals = avg_func_evals/avg_trials

print('\n\n avg_func_evals = ', avg_func_evals)


# In[ ]:


# plot solution space in space time 
fig = plt.figure(figsize=(16,4), )
ax = fig.add_subplot(111, projection='3d')

# Plot a few snapshots.
color = iter(plt.cm.viridis(np.linspace(.25, 1, 5)))

cdata = ax.scatter(X_all, Z_all, Theta_all, c=values, cmap="magma") #, shading="nearest", 
plt.colorbar(cdata, ax=ax, extend="both")
ax.set_xlabel(r"Hor. Disp. $x$")
ax.set_ylabel(r"Vert. Disp. $z$")
ax.set_zlabel(r"Orientation: $\theta$")

fig.suptitle("Dynamics")
plt.show()


# In[26]:


(torch.randn(100, 1000) + torch.randn(1000, 3).view(-1, 1)).shape 


# In[ ]:




