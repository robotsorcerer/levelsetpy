### **LevelSetsPy Library**

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

### **Installation and Prerequisites**

### **Dependencies** 

+ It's best to create a virtual or conda environment in python 3.7+ (I used Python 3.8/3.9).

+ All the dependencies listed below are installable via: `pip install -r requirements.txt`

| Dependency      | Dependency      | Dependency      | 
| :--:     | :---:               | :---:               | 
| [Numpy](https://numpy.org/)  | [Scipy](https://scipy.org/)  | [Cupy](https://cupy.dev/) |
| [absl-py](https://abseil.io/docs/python/quickstart)   | [Scikit-image](https://scikit-image.org/) | [Matplotlib](https://matplotlib.org/)     | 

### **Build and Install**

Be sure to activate your `conda` or `virtualven` environment first e.g. if your environment is named `py39`:

```bash
	conda activate py39
```

Then install to the activated environment as follows

+ BUILD: `python setup.py build --build-lib=/path/to/your-desired/build`

+ INSTALL: via setup.py: `python setup.py install`; or via pip: `pip install -e . `

Some examples listed in the [Examples](/Examples) folder are listed below:

### **Examples**

#### **Approximate Barrier Surface of the Target Tube of Two Rockets on a Plane.**

We adopt the rocket launch problem of [Dreyfus](https://apps.dtic.mil/sti/citations/AD0644592). The goal is to launch a rocket in fixed time to a desired altitude, given a final vertical velocity component and a maximum final horizontal component as constraints. For all intents and purposes pertaining to a reachability problem, we shall amend the original problem, discard the constraints and turn the problem to a planar differential game between the two rockets -- with one serving as a pursuer, $P$ and the other as an evader, $E$. Ours is an instance of the problem of Mayer. A single rocket's motion is dictated by the following differential equations (under Dreyfus' assumptions):

```math
	\begin{align}
	\dot{x}_{1} &= x_{3}; \,\, &x_{1}(t_0) = 0;
	\\
	\dot{x}_{2} &= x_{4},\,\, &x_{2}(t_0)= 0;   
	\\
	\dot{x}_{3} &= a \cos u,\, &x_{3}(t_0)= 0;
	\\
	\dot{x}_{4} &= a \sin u - g,\,\, &x_{4}(t_0)= 0
	\end{align}
```

where, $(x_1, x_2)$ are respectively the horizontal and vertical range of the rocket (in feet), $(x_3, x_4)$ are respectively the horizontal and vertical velocities of the rocket (in feet per second), while $a$ and $g$ are respectively the acceleration and gravitational accelerations (in feet per square second).

Being a free endpoint problem, we transform it into a game between two players  without the terminal time constraints  defined in Jacobson and Mayne's DDP Book. Let the states of $P$ and $E$ be now denoted as $(x_p, x_e)$ respectively which are driven by their thrusts $(u_p, u_e)$ respectively in the $xz$-plane. The relevant kinematic equations are given above.

We now make the problem amenable to a two-player differential game analysis so that every max and min operations are in the interior and no sudden changes from extremes are too aggravating in cost.

<div align="center">
<img src="Figures/rocket.jpg" height="465px" width="512px"/>
</div>

Motion of two rockets on a Cartesian $xz$-plane with a thrust inclination in relative coordinates given by $\theta:=u_p- u_e$.
Therefore, we rewrite Dreyfus's equation with ${P}$'s motion relative to ${E}$'s  along  the $(x,z)$ plane so that the relative orientation as illustrated above is $\theta=u_p- u_e$. The coordinates of ${P}$ are freely chosen; however, the coordinates of ${E}$ are chosen a distance $r$ away from $(x,z)$ so that the ${E} {P}$ vector's inclination measured counterclockwise from the $x$ axis is $\theta$. Following the conventions in the figure above, the game's relative equations of motion in _*reduced space*_ is $\mathcal{X} = (x, z, \theta)$ where $\theta \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right)$ and $(x,z) \in \mathbb{R}^2$ are

```math
	\begin{align}
	\dot{x} &= a_p \cos \theta + u_e x, \\
	\dot{z} &=a_p \sin \theta + a_e + u_e x - g, \\
	\dot{\theta} &= u_p -u_e.
	\end{align}
```

The payoff, $\Phi$, is the distance of $P$ from ${E}$ when capture occurs denoted as $\|{P} {E}\|_2$. Capture occurs when $\| {P} {E} \|_2 \le r$ for a pre-specified capture radius, $r>0$. In the equation above,  we say ${P}$ controls $u_p$ and is minimizing $\Phi$, and ${E}$ controls $u_e$ and is maximizing $P$. The boundary of the _usable part_ of the origin-centered circle of radius $r$ (we set $r=1.5$ feet in our evaluations) is $\|{P} {E}\|_2 $. In this sentiment, we find that

```math
	\begin{align}
	r^2 &=  x^2 + z^2,
	\end{align}
```

and all capture points are specified by  useable part's interior is

```math
\begin{align}
\dot{r}(x,t) + \min \left[0, H(x, \frac{\partial r(x, t)}{\partial x})\right] \le 0,
\end{align}
```

with the corresponding Hamiltonian

```math
\begin{align}
H(x, p) = -\max_{u_e \in \mathcal{U}_e} \min_{u_p \in \mathcal{U}_p
} \begin{bmatrix}
p_1 \\ p_2 \\ p_3
\end{bmatrix}^T
\begin{bmatrix}
a_p \cos \theta + u_e x \\
a_p \sin \theta + a_e + u_p x - g \\
u_p -u_e
\end{bmatrix}.
\end{align}
```

Suppose that ${E}$'s maximizing control i.e. $u_e$ is $\bar{u}_e$ and that ${P}$'s minimizing control i.e. $u_p$ is $\bar{u}_p$. We have at the point of slowest-quickest descent on the capture surface, that

```math
	\begin{align}
	\bar{u}_e &= p_1 x - p_3, \\
	\bar{u}_p &= p_3 - p_2 x.
	\end{align}
```

<div align="center">
<img src="Figures/rocket_zerolev.jpg" height="330px" width="330px"/>
<img src="Figures/rocket_ls_final.jpg" height="330px" width="330px"/>
</div>

Initial and final backward reachable tubes for the rocket system computed using the method outlined in \cite{Crandall1984, OsherFronts}. We set $a_e = a_p = 64ft/sec^2$ and $g=32 ft/sec^2$ as in Dreyfus' original example. We compute the reachable set by optimizing for the paths of slowest-quickest descent in the equation above.


We set the linear velocities and accelerations equal to one another i.e. $u_e = u_p$ and $a_e = a_p$ so that the Hamiltonian takes the form

```math
\begin{align}
H(x, p) &= -\cos(u) |a p_1| + \cos(u) |a p_1| -\sin (u) |a p_2| -  \sin (u) | ap_2 | + u | p_3| - u |p_3|.
\label{eq:rocket_hamfunc}
\end{align}
```

Using our levelsetpy toolbox, we compute the backward reachable tube of the game over a time span of $[-2.5, 0]$ seconds by running a game between the two players over 11 global optimization time steps. The initial value function (left inset of the figure above) is represented as a dynamic implicit surface over all point sets in the state space (using a signed distance function) for a coordinate-aligned cylinder whose vertical axes runs parallel to the orientation of the rockets depicted. A three-dimensional grid with uniformly spaced dimensions over an interval $[-64, 64]$ and at a resolution of $100$ points per dimension was used in updating the values. The Hamiltonian was resolved with a second-order essentially non-oscillating (ENO) upwinding scheme. This is implemented as _*upwindFirstENO2*_ function under our _*SpatialDerivative*_ package. As with all matters involving the numerical discretization schemes employed for solving Hamilton-Jacobi equations, the stability of the ensuing solution to tthe HJ PDE is of eminence. we employed a global **Lax-Friedrichs** scheme together with a total variation diminishing Runge-Kutta discretization scheme based on fluxes are chosen. The final BRT at the end of the optimization run is shown in the right inset of the figure.

#### **Time to Reach Problems**

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

A point $(x_1, x_2)$ on the state grid belongs to the set of states $S(t^\star)$ from which it can be forced to the origin $(0, 0)$ in the same minimum time $t^\star$. We call the set $S(t^\star)$ the minimum **isochrone**. These are the isochrones of the system -- akin to the isochrone map used in geography, hydrology, and transportation planning for depicting areas of equal travel time to a goal state. The level sets of the time optimal control equation correspond to the **isochrones** of the system as illustrated below

<div align="center">
<img src="Figures/Dint/isochoner_above.jpg" height="220px" width="260px"/>
<img src="Figures/Dint/isochoner_below.jpg" height="220px" width="260px"/>
<img src="Figures/Dint/isochoner_all.jpg" height="220px" width="260px"/>
</div>

+ <center><b>Left to right: L-R: (a) Isochrones for states above the switching curve, (b) states below the switching curve, (c) all states that constitute the analytic isochrones.</b></center>

<div align="center">
<img src="Figures/Dint/dint_0.jpeg" height="350px" width="750px"/>
</div>

+ <center><b>Time to reach the origin at different integration steps. Left: Closed-form Solution to the time to reach the origin problem. Right: Lax-Friedrichs Approximation to Time to Reach the Origin.</b></center>

<div align="center">
<img src="Figures/Dint/dint_02.jpeg" height="300px" width="400px"/>
<img src="Figures/Dint/dint_03.jpeg" height="300px" width="400px"/>
</div>

Time to reach the origin at different integration steps. Left: Stacked numerical BRS at $t=0.25$ secs. Right: Stacked numerical BRS at $t=0.75$ secs.


#### **Geometry of Implicit Surfaces**

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




### Citing this work

If you have found this library of routines and packages useful, please cite it:

```
@article{LevPy,
title   = {LevelSetPy: A GPU-Accelerated Python Software Package for Resolving Hamilton-Jacobi PDEs and Level Set Evolutions},
publisher={Molu, Lekan},
author  = {Molu, Lekan},
howpublished = {\url{https://github.com/robotsorcerer/LevelSetPy}},
note = {Accessed March 31, 2023}
year    = {2023},
}
```
