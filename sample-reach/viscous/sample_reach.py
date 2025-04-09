import torch 
from torch.utils.data import Dataset 

import numpy as np

import time 
import logging 
import argparse 
import sys, os
from os.path import join, expanduser 
import scipy.ndimage as sp_ndimage
from rockets import RocketDynamics

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Moreau Reachability Analysis')
parser.add_argument('--verbose', '-vb', action='store_true', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_false', help='save BRS/BRT at end of sim' )
parser.add_argument('--adapt_time', '-at', action='store_false', help='Make time adfaptive' )
parser.add_argument('--visualize', '-vz', action='store_true', help='visualize level sets?' )
parser.add_argument('--plot', '-lb', action='store_true', help='plot initial values?' )
# parser.add_argument('--trials', '-tr', type=int, default=50, help='Code seed.' )
parser.add_argument('--resolution', '-re', type=int, default=100, help='State space resolution.' )
parser.add_argument('--time_upper', '-tu', type=int, default=1, help='Upper bound of the time interval.' )
parser.add_argument('--spatial_bound', '-sb', type=int, default=64, help='Upper bound of the time interval.' )
parser.add_argument('--seed', '-sd', type=int, default=123, help='Code seed.' )
parser.add_argument('--num_samples', '-sa', type=int, default=100, help='Number of Gaussian Samples ber Batch.' )
parser.add_argument('--data_dir', '-dd', type=str, default="/opt/hj_reach", help='Number of Gaussian Samples ber Batch.' )
parser.add_argument('--num_trials', '-nt', type=int, default=50, help='Number of Gaussian Samples ber Batch.' )
parser.add_argument('--pause_time', '-pz', type=float, default=.3, help='pause time between successive updates of plots' )
args = parser.parse_args()

print(args)

if args.verbose:
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
else:
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

## Problem Statement and Numerical Setting

# Let $\Omega = [0,L]\subset \mathbb{R}$ be the spatial domain indicated by the variable $\omega$, and let $(0,T]\subset\mathbb{R}$ be the time domain with variable $t$.
# We consider the three-dimensional HJ equation with homogeneous Dirichlet boundary conditions,
# $$
# \begin{align*}
#     &\partial \bm{v}_t(t; \bm{x}) + \min\{0, \bm{H}\left(x, \partial_{\bm{x}} \bm{v}(t; \bm{x})\right)\} = 0, \qquad &\bm{v}(0; \bm{x}) = \bm{g}(0; \bm{x}) \nonumber \\
# %\bm{v}(0; \bm{x}) &= \bm{g}(0; \bm{x}) \\
# %
# &\approx \partial_t \bm{v}^\delta(t; \bm{x}) + \min\{0, 
# \bm{H}^\delta\left(t; x, \partial_{\bm{x}} \bm{v}^\delta\right)\} = 0, \qquad &\bm{v}^\delta(0; \bm{x}) = \bm{g}(0; \bm{x}) \nonumber \\
# %
# &:= \partial_t \bm{v}^\delta(t; \bm{x}) + \min\bigg\{0, 
# \max_{\bm{u} \in \mathcal{U}} \min_{\bm{w} \in \mathcal{W}} \, \bigg\langle f(t; \bm{x}, \bm{u}, \bm{w}), \frac{1}{t}(\bm{x} - \text{prox}_{t\bm{g}}(\bm{x})) \bigg\rangle
# \bigg\} = 0 \qquad &\bm{v}^\delta(0; \bm{x}) = \bm{g}(0; \bm{x}).
# \end{align*}
# $$

# This is a model for a one-dimensional rod that conducts heat: the temperature at the ends of the rod are fixed at $0$ and heat is allowed to flow out of the rod through the ends.


class HJ_MAD:
    ''' 
        Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) is used to solve nonconvex minimization
        problems via a zeroth-order sampling scheme.
        
        Inputs:
          1)  dynamics     = Class that contains the dynamics of the agents, hamiltonian function and other auxiliary variables.
          2)  delta        = coefficient of viscous term in the HJ equation
          3)  int_samples  = number of samples used to approximate expectation in heat equation solution
          4)  t_span       = time vector containig [initial time, minimum time allowed, maximum time]
          5)  max_iters    = max number of iterations
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
    def __init__(self, dynamics, delta=0.1, int_samples=100, t_span = [0, 1], max_iters=5e4, 
                 tol=5e-2, psi=0.3, beta=0.9, alpha=1.0, adapt_time=False, verbose=True):   
      self.delta            = delta
      self.g                = dynamics.get_values
      self.int_samples      = int_samples
      self.max_iters        = max_iters
      self.tol              = tol
      self.t_span           = t_span
      self.psi              = psi
      self.beta             = beta 
      self.alpha            = alpha 
      self.adapt_time       = adapt_time
      self.verbose          = verbose

      # samples tools
      self.states           = dynamics.state_space
      self.probs            = torch.ones(states.shape[0], dtype=torch.float) / states.shape[0]
      
      'HJI Hyperparameters.'
      '--------------------'
      eps = sys.float_info.epsilon
      self.t_steps = (t_span[-1] - t_span[0]) / 100
      self.small = 100 * eps 

      self.dim = states.shape[1] 
      # check that alpha is in right interval
      assert(alpha >= 1-np.sqrt(0.9))
      assert(alpha <= 1+np.sqrt(1.1))
    
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
      
      hamiltonian = dynamics.hamiltonian(grad_vk, x)
      hamterm = torch.minimum(torch.Tensor([0]), hamiltonian)

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
      xk_hist               = [torch.Tensor([0])]  
      xk_error_hist         = [] 
      rel_grad_vk_norm_hist = [] 
      gk_hist               = [1e9] 
      tk_hist               = [] 
      counter               = 0
      hji_rcbrt_term_hist   = []

      t_now                 = self.t_span[0]

      sample_idces = self.probs.multinomial(self.int_samples, replacement=True)
      xk     = self.states[sample_idces, :]
      x_opt = xk

      first_moment, _, hji_rcbrt_term   = self.compute_grad_vk(xk, t_now, self.g, self.delta)
      rel_grad_vk_norm      = 1.0

      fmt = '[{:3d}] |  t_now = {:2.4f} | gk = {:3.4f} | xk_err = {:3.4f} '
      fmt += ' | |grad_vk| = {:3.4f} | hj_term = {:2.2f} '

      print('\n')
      logger.info('-------------------------- RUNNING HJ-MAD ---------------------------')
      logger.info(f'dimension = f{self.dim}, n_samples = {self.int_samples}')

      converged = True
      while converged and (self.t_span[1]  - t_now) > self.small * self.t_span[1]:  
        
        xk_norm = torch.norm(xk, p=2, dim=0)
        xk_hist.append(xk_norm)

        rel_grad_vk_norm_hist.append(rel_grad_vk_norm)
        xk_error_hist.append(torch.norm(xk_hist[-1]-xk_hist[-2], p=2).item())
        tk_hist.append(t_now)

        func_eval = self.g(xk)
        gk_hist.append(torch.norm(torch.norm(func_eval,  2, dim=0), p=2).item())
        hji_rcbrt_term_hist.append(torch.norm(hji_rcbrt_term, p=2, dim=0).item())

        counter += 1
        t_now = t_now + self.t_steps

        if self.verbose:
          print(fmt.format(counter, t_now, gk_hist[-1], torch.norm(xk_hist[-1], p=2).item(), rel_grad_vk_norm_hist[-1], hji_rcbrt_term_hist[-1]))

			  # # How far to step?
        # self.t_span = np.hstack([t_now, min(self.t_span[1], t_now + self.t_steps)])

        if np.all(np.all(np.abs(xk_error_hist[:-3]))<self.tol):
          tk_hist = tk_hist[0:]
          xk_hist = xk_hist[0:]
          xk_error_hist = xk_error_hist[0:]
          rel_grad_vk_norm_hist = rel_grad_vk_norm_hist[0:]
          gk_hist               = gk_hist[0:]
          logger.info('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_vk_norm_hist[-1]))
          logger.info(f'iter = ', t_now, ', number of function evaluations: {len(xk_error_hist)*self.int_samples}')
          converged = False  

        if counter>10 and (gk_hist[-1] < gk_hist[-2]) and (gk_hist[-2] < gk_hist[-3]):
          x_opt = xk 

        xk -= self.alpha * first_moment 
        
        grad_vk, vk, hji_rcbrt_term = self.compute_grad_vk(xk, t_now, self.g, self.delta)

        if  self.adapt_time:
          t_now = self.update_time(t_now, rel_grad_vk_norm)

        grad_vk_norm_old = torch.norm(first_moment)
        first_moment  = self.beta*first_moment + (1-self.beta)*grad_vk

        grad_vk_norm = torch.norm(first_moment)
        rel_grad_vk_norm = grad_vk_norm/(grad_vk_norm_old + 1e-12)

      return x_opt, xk_hist, xk_error_hist, tk_hist, rel_grad_vk_norm_hist, gk_hist


def plot_values(states, title="Initial values", fname=None, fontdict={'fontsize':16, 'fontweight':'bold'}):  
  X, Z, θ = states[:,0], states[:,1], states[:,2]
  X, Z, θ =  torch.meshgrid(*(X, Z, θ ), indexing='ij') 

  a = 32; g=32; u=1; 
  values =  torch.sqrt(a * torch.cos(θ)**2  + (a * torch.sin(θ) + \
                                     a + u * X - g)**2)
  # plot solution space in space time 
  fig = plt.figure(figsize=(16,9), )
  ax = fig.add_subplot(111, projection='3d')
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
  ax.axes.get_zaxis().set_ticks([])

  cdata = ax.scatter(X, Z, θ, c=values, cmap="magma") 
  plt.colorbar(cdata, ax=ax, extend="both", shrink=0.5)
  ax.set_xlabel(r"Horz. $x$ (ft)", fontdict=fontdict)
  ax.set_ylabel(r"Vert. $z$ (ft)", fontdict=fontdict)
  ax.set_zlabel(r"Orientation: $\theta$ (rad)", fontdict=fontdict)
  ax.set_title(title, fontdict=fontdict)

  fig.suptitle("Rockets Relative Dynamics' Values", fontsize=16)
  if args.visualize:
    plt.show()
  plt.savefig(fname, bbox_inches='tight',facecolor='None', dpi=76)


def main(dynamics, resolution=1000, seed=123):  

  if args.plot:
    save_dir = join(expanduser("~"), "Documents/Papers/MSRYeatrs/ProxSampReach/figures")
    save_dir = join(expanduser("~"), "Downloads") 
    fname = join('init_values.jpg')
    plot_values(dynamics.state_space, fname=fname)

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

  max_iters       = int(5e4)

  eps= sys.float_info.epsilon
  t_span = [0+eps, 1.0]
  hj_mad_algo = HJ_MAD(dynamics, delta=0.1, int_samples=args.num_samples, t_span = t_span, max_iters=max_iters, 
                                      tol=5e-2, psi=0.1, beta=0.9, alpha=1.0, adapt_time=True, verbose=True)

  # run 30 times 
  avg_func_evals  = 0

  x_opt_list, xk_hist_list, tk_hist_list, xk_error_hist_list, \
      rel_grad_uk_norm_hist_list, globalk_hist_list = [[np.nan for x in range(args.num_trials)]]*6
  
  # no elems
  x_opt_list, xk_hist_list, tk_hist_list, xk_error_hist_list, \
      rel_grad_uk_norm_hist_list, globalk_hist_list = [[]]*6
  
  if not os.path.exists(join(args.data_dir, args.experiment)):
    os.makedirs(join(args.data_dir, args.experiment))

  for trial in range(args.num_trials):
    print(f">>>Rolling on sample trial {trial}/{args.num_trials}.")
    x_opt, xk_hist, tk_hist, xk_error_hist, \
      rel_grad_uk_norm_hist, globalk_hist = hj_mad_algo.run()
    
    # stack em results
    x_opt_list += x_opt 
    xk_hist_list += xk_hist
    tk_hist_list += tk_hist
    xk_error_hist_list += xk_error_hist
    rel_grad_uk_norm_hist_list += rel_grad_uk_norm_hist
    globalk_hist_list += globalk_hist

    avg_func_evals += len(xk_error_hist)*args.num_samples

  avg_func_evals = avg_func_evals/args.num_trials

  print('\n avg_func_evals = ', avg_func_evals)

if __name__ == "__main__":
  
  resolution = 100; L = 100
  dynamics = RocketDynamics(1, 1, T=args.time_upper, L=args.spatial_bound, a=32, g=32, resolution=args.resolution)
  states =  dynamics.state_space
  main(dynamics, resolution, seed=args.seed)


# python sample_reach.py --num_trials 1 --num_samples 100/150/200/250
