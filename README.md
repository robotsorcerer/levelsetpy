### **LevelSetsPy Library**

#### BLUF - May 2024. 

**What?** GPU-accelerated software package for solving initial value  hyperbolic partial differential equations, particularly of the evolution form of Cauchy-type Hamilton-Jacobi (HJ) equations. These HJ equations are increasingly attracting attention in the control community for analyzing reachability problems in robotics, transport, biology and other problem domains of late.  
**Focus:** Safety-critical analysis of automated systems in optimal control and reachability settings. 
**Plug:** Easy portability and extensibility to modern libraries for the analyses of safety-critical (reinforcement) learning, control, robotics, transport, and flow problems among others.

<div align="center">
 <img src="figures/rocket.jpg" height="250px" width="250px">
 <img src="figures/rocket_zerolev.jpg" height="250px" width="250px">
 <img src="figures/rocket_ls_final.jpg" height="250px" width="250px">
</div>


### **Installation and Prerequisites**

### **Dependencies** 


| :--:     | :---:               | :---:               | 
| [Numpy](https://numpy.org/)  | [Scipy](https://scipy.org/)  | [PyTorch](https://pytorch.org/) |
| [Absl-py](https://abseil.io/docs/python/quickstart)   | [Scikit-image](https://scikit-image.org/) | [Matplotlib](https://matplotlib.org/)     | 


### **Build and Install**

Activate your `conda` or `virtualven` environment, then install as follows

```bash
python setup.py build --build-lib=/path/to/your-desired/build
pip install -e . 
```


A separate `README.md` file is left in the respective folders.

### **Examples**

#### **Initial Value Problems (IVP)**

+ Making a 2D Grid 

```python

from levelsetpy.grids import *
import matplotlib.pyplot as plt
from levelsetpy.utilities import *
from levelsetpy.boundarycondition import *
from levelsetpy.visualization import *
from math import pi
from levelsetpy.initialconditions import *


# ### A Basic 2-D Grid and a signed distance function cylinder

delay = 1
block=False
fontdict = {'fontsize':12, 'fontweight':'bold'}

from math import pi
gridMin = np.array([[0,0]])
gridMax = np.array([[5, 5]])
N = 20 *np.ones((2,1)).astype(np.int64)
g = createGrid(gridMin, gridMax, N, low_mem=False, process=True)

savedict = dict(save=True, savename='2d_grid.jpg', savepath=join("..", "jpeg_dumps"))
viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
viz.visGrid([g], g.dim, title='Simple 2D Grid')
```

 - Spheres on a 2D grid: a single sphere, two spheres, union of two spheres, and difference of a sphere and a rectangle.

<!-- <div align="center">
 <img src="figures/shapes2d/sphere_2d.jpeg" height="250px" width="250px">
 <img src="figures/shapes2d/sphere2_2d.jpeg" height="250px" width="250px">
 <img src="figures/shapes2d/sphere_union_2d.jpeg" height="250px" width="250px">
 <img src="figures/shapes2d/sph_rect_diff_2d.jpeg" height="250px" width="250px">
</div>

- Rectangles on a 2D grid: a single rectangle, two rectangles, union of two rectangles, and a cylinder.

<div align="center">
 <img src="figures/shapes2d/rect3_2d.jpeg" height="250px" width="250px">
 <img src="figures/shapes2d/rect4_2d.jpeg" height="250px" width="250px">
 <img src="figures/shapes2d/rect_union_2d.jpeg" height="250px" width="250px">
 <img src="figures/shapes2d/cylinder_2d.jpeg" height="250px" width="250px">
</div> -->
<p align="center"><b>Spheres and Sphere Operations on a 2D Grid</b></p>

<table align="center">
  <tr>
    <th align="center">Single Sphere</th>
    <th align="center">Second Sphere</th>
    <th align="center">Union of Two Spheres</th>
    <th align="center">Sphere-Rectangle Difference</th>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/shapes2d/sphere_2d.jpeg" width="250">
    </td>
    <td align="center">
      <img src="figures/shapes2d/sphere2_2d.jpeg" width="250">
    </td>
    <td align="center">
      <img src="figures/shapes2d/sphere_union_2d.jpeg" width="250">
    </td>
    <td align="center">
      <img src="figures/shapes2d/sph_rect_diff_2d.jpeg" width="250">
    </td>
  </tr>
</table>

<br>

<p align="center"><b>Rectangles and Cylinders on a 2D Grid</b></p>

<table align="center">
  <tr>
    <th align="center">Single Rectangle</th>
    <th align="center">Two Rectangles</th>
    <th align="center">Union of Two Rectangles</th>
    <th align="center">Cylinder</th>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/shapes2d/rect3_2d.jpeg" width="250">
    </td>
    <td align="center">
      <img src="figures/shapes2d/rect4_2d.jpeg" width="250">
    </td>
    <td align="center">
      <img src="figures/shapes2d/rect_union_2d.jpeg" width="250">
    </td>
    <td align="center">
      <img src="figures/shapes2d/cylinder_2d.jpeg" width="250">
    </td>
  </tr>
</table>

### 3D Grids

- Initial Conditions for a 3D Grid: a sphere, a cylinder, a sphere and a cylinder difference, a sphere and a cylinder union, and a sphere and a an iconosphere. Check out the [source code](levelsetpy/tests/test_mesh_3d.py).


<p align="center"><b>3D Shapes and Set Operations</b></p>

<table align="center">
  <tr>
    <td align="center">
      <img src="figures/shapes3d/sphere.jpeg" width="250"><br>
      Sphere
    </td>
    <td align="center">
      <img src="figures/shapes3d/cylinder.jpeg" width="250"><br>
      Cylinder
    </td>
    <td align="center">
      <img src="figures/shapes3d/sphere_cyl_diff.jpeg" width="250"><br>
      Difference
    </td>
    <td align="center">
      <img src="figures/shapes3d/sphere_cyl_union.jpeg" width="250"><br>
      Union
    </td>
    <td align="center">
      <img src="figures/shapes3d/sphere_cyl_intersect.jpeg" width="250"><br>
      Intersection
    </td>
  </tr>
</table>

#### Robustly Controlled Backward Reachable Tubes (RCBRTs)



+ RCBRT of a [2D Rocket System](levelsetpy/examples/rocket_ls_example.py) for two rockets on a 2D plane: initial zero level set, and the final RCBRT.

<div align="center">
 <img src="figures/rocket.jpg" height="250px" width="250px">
 <img src="figures/rocket_zerolev.jpg" height="250px" width="250px">
 <img src="figures/rocket_ls_final.jpg" height="250px" width="250px">
</div>


+ The [RCBRT of a Dubins Car System on a Plane](levelsetpy/examples/dubins_ls_example.py)


+ [Time to reach the target set for a double integrator on a plane](levelsetpy/examples/dint_basic.py). Double integrator on a plane with the analytical time to reach the target set: Switching Curve. Initial Conditions.

<div align="center">
 <img src="figures/dint/attr.jpg" height="200px" width="250px">
 <img src="figures/dint/switching_curve.jpg" height="200px" width="250px">
 <img src="figures/dint/doub_int_trajos.jpg" height="200px" width="250px">
</div>

+ Double integrator on a plane: Isochoner above switching curve. Isochoner below switching curve. Isochoner above and below switching curve. 

<div align="center">
 <img src="figures/dint/isochoner_above.jpg" height="200px" width="250px">
 <img src="figures/dint/isochoner_below.jpg" height="200px" width="250px">
 <img src="figures/dint/isochoner_all.jpg" height="200px" width="250px">
</div>



---

## HJ-Gauss full residual computations

The research code in [`levelsetpy/monte_carlo/`](levelsetpy/monte_carlo/) retains the complete generalized Cole–Hopf residual. It is organized as `hjgauss/core`, `hjgauss/validation`, and `hjgauss/render`, with tests under `tests/`. It includes transformed-equation solvers, a supplied-source Gaussian Duhamel estimator, analytic-solution checks, and reproducible multidimensional validation workflows.

See the [implementation README](levelsetpy/monte_carlo/README.md) for commands, provenance checks, and numerical scope. Generated native fields and reports live under the ignored `results/` directory and are recreated by the validation commands.

The separate [`levelsetpy/mapf/`](levelsetpy/mapf/) project retains its independent grid-reference precomputation. Its sampler-dependent precomputation requires a full residual implementation and currently raises an explicit error.

```sh
cd levelsetpy/monte_carlo
python -m pip install -e '.[dev]'
python -m pytest -q
python -m hjgauss.validation.benchmark
```

---

### Citing this work

If you have found this library of routines and packages useful, please cite it:

```
@article{LevPy,
title   = {LevelSetPy: A GPU-Accelerated Python Software Package for HJ Reachability Analysis and Level Set Evolutions.},
publisher={ACM Transactions on Mathematical Software},
author  = {Molu, Lekan},
howpublished = {\url{https://github.com/robotsorcerer/levelsetpy}},
note = {Accessed July 31, 2026}
year    = {2026},
}

@inproceedings{molu2024python,
  title={The Python LevelSet Toolbox (LevelSetPy)},
  author={Molu, Lekan},
  booktitle={2024 IEEE 63rd Conference on Decision and Control (CDC)},
  pages={8938--8945},
  year={2024},
  organization={IEEE}
}

@article{HJGauss,
      title={HJ-Gauss: A Monte-Carlo HJ Reachability Scheme}, 
      author={Molu, Lekan and Renganathan, Venkatraman and Cho, Namhoon},
      year={2026},
      eprint={2605.18566},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2605.18566}, 
      howpublished = {\url{https://github.com/robotsorcerer/levelsetpy/tree/main/levelsetpy/monte_carlo}},
}
```
