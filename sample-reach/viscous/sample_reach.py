

__copyright__ 	= "2025, Hamilton-Jacobi Analysis in Python"
__comment__     = "Adaptive gradient descent on the Moreau envelope of the viscous HJ value function."
__credits__  	  = "Haoxiang You, Ian Abraham."
__license__ 	  = "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		  = "lekanmolu@microsoft.com"
__status__ 		  = "Completed"

import torch 
import numpy as np

import time 
import logging 
import argparse 
import sys, os
from datetime import datetime 
from rockets import RocketDynamics
from hj_adaptive_descent import HJ_MAD
from os.path import join, expanduser 

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
parser.add_argument('--data_dir', '-dd', type=str, default="/opt/", help='Number of Gaussian Samples ber Batch.' )
parser.add_argument('--experiment', '-ex', type=str, default="prox_reach", help='Number of Gaussian Samples ber Batch.' )
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
  hj_mad_algo = HJ_MAD(dynamics, delta=0.1, int_samples=args.num_samples, t_span = t_span, tol=5e-2, 
                                psi=0.01, beta=0.9, alpha=1.0, adapt_time=True, verbose=True)

  # run 30 times 
  avg_func_evals  = 0

  # no elems
  x_opt_list, xk_hist_list, tk_hist_list, xk_error_hist_list, \
      rel_grad_uk_norm_hist_list, gk_hist_list = [[]]*6
  
  save_path = join(args.data_dir, args.experiment)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  for trial in range(args.num_trials):
    print(f">>>Rolling on sample trial {trial}/{args.num_trials}.")
    x_opt, xk_hist, tk_hist, xk_error_hist, \
      rel_grad_uk_norm_hist, gk_hist = hj_mad_algo.run()
    
    # # stack em results
    # x_opt_list += x_opt 
    # xk_hist_list = np.hstack(xk_hist)
    # tk_hist_list += tk_hist
    # xk_error_hist_list += xk_error_hist
    # rel_grad_uk_norm_hist_list += rel_grad_uk_norm_hist
    # gk_hist_list += gk_hist

    avg_func_evals += len(xk_error_hist)*args.num_samples

    fdir = join(save_path, datetime.strftime(datetime.now(), '%m_%d_%y-%H_%M_%S'))
    os.makedirs(fdir, exist_ok=True)

    fname = join(fdir, f'trial_{trial:0>1}_evals_{avg_func_evals}.npz')
    print(f'Saving to {fname}')
    # print([type(A) for A in (tk_hist, xk_hist xk_error_hist[-1], rel_grad_uk_norm_hist[-1], gk_hist[-1])])
    np.savez_compressed(fname, t_hist=np.asarray(tk_hist), x_hist=xk_hist,  
                        delta_x=np.asarray(xk_error_hist), heat_kernel=np.asarray(rel_grad_uk_norm_hist), value_func=np.asarray(gk_hist))
    

  avg_func_evals = avg_func_evals/args.num_trials

  print('\n avg_func_evals = ', avg_func_evals)

if __name__ == "__main__":
  
  resolution = 1000; L = 100
  dynamics = RocketDynamics(1, 1, T=args.time_upper, L=args.spatial_bound, a=32, g=32, resolution=args.resolution)
  states =  dynamics.state_space
  main(dynamics, resolution, seed=args.seed)


# python sample_reach.py --num_trials 1 --num_samples 100/150/200/250
