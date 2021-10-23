### LevelSets Py

This repo is a reproduction of Ian Mitchell's LevelSets Toolbox in Python written by Lekan Molu in August, 2021.

### Demos

The [Tests](/Tests) folder contains examples of running basic tests. For example, you can view a demo of the grids creation by running

`python Tests/test_grids.py`

### Tutorial

```
python .\tutorial.py --verbose
```

CMD argument options can be passed as

```
+ verbose: False/True. How much debug info to print?
+ visualize: True, 'Show plots?'
+ pause_time: .1, 'Time to pause between updating the plots during HJ Optimization'
+ order_type: 'F', 'Use Fortran-order ('F')  or C-order ('C') for array indexing and memory storage.'
+ hj_progress, False, 'Display optimization progress'
```

### TODO's

+ Bugs in implementation of `upwindFirstWENO5.py`.
+ Interactive display of the zero sublevel set of the value function during the optimization of trajectories for Sylvia's example in the [tutorial](/tutorial.py) file.
+ My implementation of Sylvia's projection operator, `data_proj`, segfaults for certain problems.

#### Reduced Order Model TODOs

   
Let's do this in a new branch we shall call ROMs.
+ Introduce the `model order reduction` module for our time-resolved separable Hilbert spatial space decomposition schemes for handling `>5` dimensional systems ().
    - Replace time-space uniform gridding/discretization with ROM operator inference scheme on toy problems.
    - Then scale!
    
Some good references:
+ [Combining deep learning and ROMs, Bharttacharya et al. Model Reduction And Neural Networks For Parametric PDEs.](https://arxiv.org/pdf/2005.03180.pdf)
+ [Lift and Learn -- Wilcox Group, Longhorn Cattle School](https://arxiv.org/pdf/1912.08177.pdf)
+ [Proper Generalized Decomposition --  basically all the references from Chinesta](https://github.com/robotsorcerer/awesome-screw-theory/blob/master/decomp.pdf)
+ [Dynamic Mode Decomposition](): Whereupon we learn a low-dimensional linear mapping based on state space data, approximating the eigenstates of an infinite-dimensional Koopman operator. I suspect this only applies to linear systems but it would be helpful to have implementations and innovations as part of the toolbox. Example papers
   - [S. L. Brunton, J. L. Proctor, J. H. Tu, and J. N. Kutz, Compressed sensing and dynamic mode decomposition, Journal of Computational Dynamics, 2 (2015), p. 165.](https://www.aimsciences.org/journals/displayArticlesnew.jsp?paperID=12620)
   - [P. J. Schmid, Dynamic mode decomposition of numerical and experimental data, Journal of Fluid Mechanics, 656 (2010), pp. 5–28.](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4)
+ [Works that incorporate DMD with active subspace strategies to reduce the dimensionality of the parameter space]()
  - [M. Tezzele, N. Demo, G. Stabile, A. Mola, and G. Rozza, Enhancing CFD predictions in shape design problems by model and parameter space reduction, Advanced Modeling and Simulation in Engineering Sciences, 7 (2020), pp. 1–19.](https://link.springer.com/content/pdf/10.1186/s40323-020-00177-y.pdf)
   - 
