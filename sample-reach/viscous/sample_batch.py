import torch 
from torch.utils.data import Dataset 

import numpy as np

import time 
import argparse 
import sys, os
from gmm import GMM
from os.path import join, expanduser 
import scipy.ndimage as sp_ndimage
from rockets import RocketDynamics

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Moreau Reachability Analysis')
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_false', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--plot', '-lb', action='store_true', help='plot initial values?' )
parser.add_argument('--benchmark', '-bm', action='store_true', help='Benchmark this computation?' )
parser.add_argument('--stochastic', '-st', action='store_true', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--compute_traj', '-ct', action='store_false', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--seed', '-sd', type=int, default=123, help='Code seed.' )
parser.add_argument('--elevation', '-el', type=float, default=5., help='elevation angle for target set plot.' )
parser.add_argument('--direction', '-dr',  action='store_true',  help='direction to grow the level sets. Negative by default.' )
parser.add_argument('--azimuth', '-az', type=float, default=15., help='azimuth angle for target set plot.' )
parser.add_argument('--pause_time', '-pz', type=float, default=.3, help='pause time between successive updates of plots' )
args = parser.parse_args()

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

class HJ_MAD:
    ''' 
        Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) is used to solve nonconvex minimization
        problems via a zeroth-order sampling scheme.
        
        Inputs:
          1)  dynamics     = Class that contains the dynamics of the agents, hamiltonian function and other auxiliary variables.
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
    def __init__(self, dynamics, x_true, delta=0.1, int_samples=100, t_vec = [1.0, 1e-3, 1e1], max_iters=5e4, 
                 tol=5e-2, psi=0.9, beta=[0.9], eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, verbose=True):
      
      self.delta            = delta
      self.g                = dynamics.get_values
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
      y = self.noise_samples(x, delta, t)
      
      exp_term = torch.exp(-g(y)/delta)
      phi_delta       = torch.mean(exp_term)

      # separate grad_v into two terms for numerical stability
      # print(f'y: {y.shape} exp_term: {exp_term.shape}')
      numerator = y.t()*exp_term 
      numerator = torch.mean(numerator.t(), dim=0)      
      grad_vk = (x.squeeze() -  numerator/(phi_delta + self.small)) #.view(-1, 1) # the t gets canceled with the update formula
      
      hamiltonian = dynamics.hamiltonian(grad_vk, x)
      hji_rcbrt_hamterm = torch.minimum(torch.Tensor([0]), hamiltonian)

      vk       = -delta * torch.log(phi_delta+self.small)

      hji_rcbrt = vk + hji_rcbrt_hamterm

      return grad_vk, vk, hji_rcbrt

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
      hji_rcbrt_term_hist   = []

      xk    = x0
      x_opt = xk
      t_now    = self.t_vec[0]
      # t_max = self.t_vec[-1]

      first_moment, _, hji_rcbrt_term   = self.compute_grad_vk(xk, t_now, self.g, self.delta)
      rel_grad_vk_norm      = 1.0

      fmt = '[{:3.4f}]: gk = {:6.2e} | xk_err = {:6.2e} | hj_term = {:2.2e} '
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
        hji_rcbrt_term_hist.append(torch.linalg.norm(hji_rcbrt_term, 2))

        if self.verbose:
          print(fmt.format(t_now, gk_hist[-1], hji_rcbrt_term_hist[-1], xk_error_hist[-1], rel_grad_vk_norm_hist[-1], t_now))

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
        elif t_now>=self.small*self.t_vec[1]:
          print('HJ-MAD failed to converge with rel grad norm {:6.2e}'.format(rel_grad_vk_norm_hist[k]))
          print('iter = ', t_now, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          print('Used fixed time = ', self.fixed_time)

        if t_now>0:
          if gk_hist[-1] < gk_hist[-2]:
            x_opt = xk 

        xk -= self.alpha * first_moment # tk gets canceled out with gradient formula
        
        grad_vk, vk, hji_rcbrt_term = self.compute_grad_vk(xk, t_now, self.g, self.delta)

        # print(f'grad_vk: {grad_vk.shape} | hji_rcbrt_term2: {hji_rcbrt_term.shape}')
        # time.sleep(40)

        if self.fixed_time == False:
          t_now = self.update_time(t_now, rel_grad_vk_norm)

        grad_vk_norm_old = torch.norm(first_moment)
        first_moment  = self.beta*first_moment + (1-self.beta)*grad_vk

        grad_vk_norm = torch.norm(first_moment)
        rel_grad_vk_norm = grad_vk_norm/(grad_vk_norm_old + 1e-12)

      return x_opt, xk_hist, xk_error_hist, tk_hist, rel_grad_vk_norm_hist, gk_hist

def main(dynamics, resolution=1000, seed=123):  

  torch.manual_seed(seed)
  np.random.seed(seed)

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

  hj_mad_algo = HJ_MAD(dynamics, x_true, delta=0.1, int_samples=int_samples, t_vec = [0, 1.0], max_iters=int(5e4), 
                  tol=5e-2, psi=0.9, beta=0.9, eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, verbose=True)

  # run 30 times 
  avg_trials      = 30
  avg_func_evals  = 0



  for i in range(avg_trials):
    x_opt, xk_hist, tk_hist, xk_error_hist, \
      rel_grad_uk_norm_hist, globalk_hist = hj_mad_algo.run(x0)
    avg_func_evals = avg_func_evals + len(xk_error_hist)*int_samples

  avg_func_evals = avg_func_evals/avg_trials

  print('\n\n avg_func_evals = ', avg_func_evals)




def plot_values(states, title="Initial values", fname=None, fontdict={'fontsize':16, 'fontweight':'bold'}):  
  X, Z, θ = states[:,0], states[:,1], states[:,2]
  # print(f'X: {X.shape} Z: {Z.shape} θ: {θ.shape}')
  X, Z, θ =  torch.meshgrid(*(X, Z, θ ), indexing='ij') 
  # print(f'X: {X.shape} Z: {Z.shape} θ: {θ.shape}')

  a = 32; g=32; u=1; 
  values =  torch.sqrt(a * torch.cos(θ)**2  + (a * torch.sin(θ) + \
                                     a + u * X - g)**2)
  # values = torch.sqrt(X * X + θ * θ)
  # plot solution space in space time 
  fig = plt.figure(figsize=(16,9), )
  ax = fig.add_subplot(111, projection='3d')
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
  ax.axes.get_zaxis().set_ticks([])
  # Plot a few snapshots.
  # color = iter(plt.cm.viridis(np.linspace(.25, 1, 5)))


  cdata = ax.scatter(X, Z, θ, c=values, cmap="magma") #, shading="nearest", 
  plt.colorbar(cdata, ax=ax, extend="both", shrink=0.5)
  ax.set_xlabel(r"Horz. $x$ (ft)", fontdict=fontdict)
  ax.set_ylabel(r"Vert. $z$ (ft)", fontdict=fontdict)
  ax.set_zlabel(r"Orientation: $\theta$ (rad)", fontdict=fontdict)
  ax.set_title(title, fontdict=fontdict)

  fig.suptitle("Rockets Relative Dynamics' Values", fontsize=16)
  # plt.show()
  plt.savefig(fname, bbox_inches='tight',facecolor='None', dpi=76)


if __name__ == "__main__":
  
  resolution = 100; L = 100
  dynamics = RocketDynamics(1, 1, T=1, L=L, a=32, g=32, resolution=resolution)
  states =  dynamics.state_space
  print(f'states: {states.shape}')

  if args.plot:
    save_dir = join(expanduser("~"), "Documents/Papers/MSRYeatrs/ProxSampReach/figures")
    save_dir = join(expanduser("~"), "Downloads") #/Papers/MSRYeatrs/ProxSampReach/figures")
    fname = join('init_values.jpg')
    plot_values(dynamics.state_space, fname=fname)
  else:
    main(dynamics, resolution, seed=args.seed)


