### LevelSetsPy Library

This codebase implements GPU-accelerated software package for solving initial value  hyperbolic partial differential equations, particularly of the evolution form of Cauchy-type Hamilton-Jacobi (HJ) equations. These HJ equations are increasingly attracting attention in the control community for analyzing reachability problems in robotics, transport, biology and other problem domains of late.  We focus on safety-critical analysis of automated systems in optimal control and reachability settings. This software package allows easy portability and extensibility to modern libraries for the analyses of safety-critical algorithms for (reinforcement) learning, control, robotics, transport, and flow problems among others.


For the technical details on the theory behind this work, please see this paper:

```
@article{LevPy,
title   = {A GPU-Accelerated Python Software Package for HJ Reachability Analysis and Level Set Evolutions.},
publisher={Molu, Lekan},
author  = {Molu, Lekan},
howpublished = {\url{https://github.com/robotsorcerer/LevelSetPy}},
note = {Accessed March 31, 2023}
year    = {2023},
}
```

#### Geometry of Implicit Surfaces

+ Implicit representation of 2D geometric primitives on a [grid](/Grids) in 2 dimensions are herewith put forward. We construct surfaces implicitly on 2D grid nodes by performing elementary geometric operations between the representation of specific geometric primitives and grid nodal points. All the codes for reproducing these geometries are available in [test_mesh_2d_only.py](Tests/test_mesh_2d_only.py). Jupyter notebook files are available in [init_conds.ipynb](Notes/init_conds.ipynb).

+ <center><b>Left to right: A cylinder, a rectangle, and a sphere.</b></center>

<div align="center">
<img src="Figures/Shapes2D/cylinder_2d.jpeg" height="330px" width="260px"/>
<img src="Figures/Shapes2D/rect4_2d.jpeg" height="330px" width="260px"/>
<img src="Figures/Shapes2D/sphere2_2d.jpeg" height="330px" width="260px"/>
</div>


+ <center><b>Left to right: Union of two spatially separated rectangles on a grid; union of two spheres; and union of a sphere and a rectangle.</b></center>

<div align="center">
<img src="Figures/Shapes2D/rect_union_2d.jpeg" height="330px" width="260px"/>
<img src="Figures/Shapes2D/sphere_union_2d.jpeg" height="330px" width="260px"/>
<img src="Figures/Shapes2D/sph_rect_diff_2d.jpeg" height="330px" width="260px"/>
</div>

+ Implicit representation of 3D geometric primitives on a [grid](/Grids) in 3 dimensions are herewith put forward. We construct surfaces implicitly on 3D grid nodes by performing elementary geometric operations between the representation of specific geometric primitives and grid nodal points. All the codes for reproducing these geometries are available in [test_mesh_3d.py](Tests/test_mesh_3d_only.py).

- <center><b>Left to right: A cylinder, and a sphere.</b></center>

<div align="center">
<img src="Figures/Shapes3D/cylinder.jpeg" height="330px" width="330px"/>
<img src="Figures/Shapes3D/sphere.jpeg" height="330px" width="330px"/>
</div>

- <center><b>Left to right: Union of a sphere and cylinder, and intersection of a sphere and cylinder.</b></center>

<div align="center">
<img src="Figures/Shapes3D/sphere_cyl_union.jpeg" height="350px" width="330px"/>
<img src="Figures/Shapes3D/sphere_cyl_intersect.jpeg" height="350px" width="330px"/>
</div>


### Time to Reach Problems

**The jupyter notebook file for reproducing the experiments in this section are available in [dint_basic.ipynb](Notes/dint_basic.ipynb).**

At issue are what point sets  in the phase space can be reached through the choice of a control input. Specifically, we consider the  double integral plant which has the following second-order dynamics

```math
\ddot{x}(t) = u(t)
```

and which admits the bounded control signals $\mid u(t) \mid \le 1$ for all time $t$. After a change of variables,we have the following system of first-order differential equations

```math
\begin{align}
\dot{x}_1(t) &= x_2(t),\quad
\dot{x}_2(t) = u(t), \quad \mid u(t) \mid \le 1. \nonumber
\end{align}
```

The reachability problem is to address the possibility of reaching all points in the state space in a **transient** manner. That is, we would like to find point sets on the state space, at a particular time step, such that we can bring the system to an equilibrium, say, $\left(0, 0\right)$ -- and once we reach equilibrium,  we would like to remain on these states for all future times.

Therefore, we set the running cost to zero, so that the Hamiltonian is
%
```math
\begin{align}
	H(x, p) = p_1 \dot{x}_1 + p_2 \dot{x}_2.
\end{align}
```

The necessary optimality condition stipulates that the minimizing control law be
```math
\begin{align}
	u(t) = -\text{ sign }(p_2(t)).
\end{align}
```

On a finite time interval, $t \in [t_0, t_f]$, the time-optimal $u(t)$ is a constant $k$ so that for initial conditions $x_1(t_0) = {\xi}_1$ and $x_2(t_0) = {\xi}_2$, it can be  verified that state trajectories obey the relation

```math
\begin{align}
x_1(t) = {\xi}_1 + \frac{1}{2} k  \left(x_2^2 - {\xi}_2^2\right), \,\text{for } t = k \left(x_2(t) - {\xi}_2\right).
\end{align}
```

Phase portrait of state trajectories is shown below

<div align="center">
<img src="Figures/doub_int_trajos.jpg" height="350px" width="680px"/>
</div>

The time to go from any point on any of the intersections to the origin on the state trajectories of the figure displayed is our approximation problem.  This minimum time admits an analytical solution given by

```math
\begin{align}
t^\star(x_1, x_2) =
\begin{cases}
x_2 + \sqrt{4 x_1 + 2 x_2^2} \, &\text{if } \, x_1 > \dfrac{1}{2} x_2 |x_2| \\
-x_2 + \sqrt{-4 x_1 + 2 x_2^2} \, &\text{if }  \, x_1 < -\dfrac{1}{2} x_2 |x_2| \\
|x_2| &\text{if } x_1 = \dfrac{1}{2} x_2 |x_2|.
\end{cases}
\end{align}
```


Let us define $R_+$ as the portions of the state space above the curve $\Omega$\footnote{The switching curve is the mid-partition of the time to reach chart with bright orange color.} and $R_-$ as the portions of the state space below the curve $\Omega$.  The confluence of the locus of points on $\Omega_+$ and $\Omega_-$ is the switching curve, depicted on the left inset of \autoref{fig:attr}, and given as

```math
\begin{align}
\Omega \triangleq \Omega_+ \cup \Omega_- &= \left\{(x_1, x_2): x_1 = \frac{1}{2}x_2 \mid x_2 \mid \right\}.
\end{align}
```


The analytical time to reach the origin, after computation (please see the cited paper above), is

<div align="center">
<img src="Figures/attr.jpg" height="350px" width="680px"/>
</div>

whereupon the switching curve is illustrated by the golden surface along the "zero-phase" of the phase plot above. We will like to use the level sets library to (over)-approximate this analytical time-to-reach the origin.



### Citing this work

If you have found this library of routines and packages useful for you, please cite it.

```
@article{LevPy,
title   = {A GPU-Accelerated Python Software Package for HJ Reachability Analysis and Level Set Evolutions.},
publisher={Molu, Lekan},
author  = {Molu, Lekan},
howpublished = {\url{https://github.com/robotsorcerer/LevelSetPy}},
note = {Accessed March 31, 2023}
year    = {2023},
}
```
