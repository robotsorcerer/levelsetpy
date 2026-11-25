# LevelSetPy Library

## Contents

- [LevelSetPy overview](#levelsetpy-overview)
- [Research implementations](#research-implementations)
  - [Cole--Hopf transformed solvers](#cole--hopf-transformed-solvers)
  - [HJ-Gauss sampling-based reachability](#hj-gauss-sampling-based-reachability)
  - [Simultaneous pairwise reachability filtering for lifelong MAPF](#simultaneous-pairwise-reachability-filtering-for-lifelong-mapf)
- [Installation and prerequisites](#installation-and-prerequisites)
- [Examples](#examples)
- [Citing this work](#citing-this-work)

## LevelSetPy overview

### BLUF — May 2024

**What?** GPU-accelerated software package for solving initial value  hyperbolic partial differential equations, particularly of the evolution form of Cauchy-type Hamilton-Jacobi (HJ) equations. These HJ equations are increasingly attracting attention in the control community for analyzing reachability problems in robotics, transport, biology and other problem domains of late.

**Focus:** Safety-critical analysis of automated systems in optimal control and reachability settings.

**Plug:** Easy portability and extensibility to modern libraries for the analyses of safety-critical (reinforcement) learning, control, robotics, transport, and flow problems among others.

<div align="center">
 <img src="figures/rocket.jpg" height="250px" width="250px">
 <img src="figures/rocket_zerolev.jpg" height="250px" width="250px">
 <img src="figures/rocket_ls_final.jpg" height="250px" width="250px">
</div>

## Research implementations

### Cole--Hopf transformed solvers

Paper: [Cole--Hopf Transformed Solvers for Viscous Hamilton--Jacobi Reachability](https://scriptedonachip.com/downloads/Papers/colehopf.pdf)

The [`levelsetpy/cole-hopf/`](levelsetpy/cole-hopf/) project studies an exact variable-coefficient exponential transformation of a viscous Hamilton--Jacobi equation. Its objective is to retain the complete algebraic and derivative residual, solve the transformed semilinear PDE deterministically, and check the recovered value field against analytic solutions and independent discretizations of the same viscous equation.

The implementation contains Fourier/exponential and finite-difference/RK4 solvers, independent same-viscosity references, refinement and boundary-sensitivity workflows, source-bound result archives, and publication renderers. The paper reports maximum transformed-versus-reference discrepancies of `5.6024e-4` for the double integrator, `1.7754e-3` for the single Dubins vehicle, `2.4724e-3` for relative Dubins, `3.2232e-3` for rockets, and `2.9126e-3` for the starling-inspired case. These are finite-grid comparisons, not continuum error bounds.

```sh
cd levelsetpy/cole-hopf
python -m pip install -e '.[dev]'
python -m pytest -q
python -m colehopf.validation.benchmark
```

### HJ-Gauss sampling-based reachability

Paper: [Sampling-based HJ Reachability](https://scriptedonachip.com/downloads/Papers/roysoc.pdf)

The [`levelsetpy/hjgauss/`](levelsetpy/hjgauss/) project implements the paper's grid-free point evaluator. It applies depth-first nonlinear Picard iteration to the transformed mild equation, estimates the heat and Duhamel terms with Gaussian samples, and uses source-time importance sampling for the score-gradient estimator. The deterministic transformation and field-validation machinery shared with the paper lives in [`levelsetpy/cole-hopf/`](levelsetpy/cole-hopf/); it is not duplicated here.

The checked-in scaling record evaluates a bounded advection--diffusion problem with a known viscous solution. At fixed Picard level two and 12 samples per update, increasing state dimension from 8 to 2048 changed median point-query runtime from `0.0437 s` to `0.0748 s` and the directly counted live coordinate payload from `0.004 MiB` to `0.938 MiB`. At dimension eight and Picard level two, increasing samples from 4 to 48 raised runtime from `0.0050 s` to `0.9103 s`; with six samples, increasing Picard level from 0 to 3 raised runtime from `0.00059 s` to `0.05296 s`. The results support grid-free point-query storage for fixed depth, not dimension-independent runtime or whole-field reconstruction.

```sh
cd levelsetpy/hjgauss
python -m pip install -e '.[dev]'
python -m pytest -q
python scaling.py --output results/scaling.json --replicates 3
```

The animations below show the shared deterministic evidence, left to right as **independent same-viscosity reference → transformed solver → absolute difference**.

<table>
  <tr>
    <th>Rockets</th>
    <th>Dubins vehicles</th>
  </tr>
  <tr>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_rockets.gif" width="440" alt="HJ-Gauss rockets comparison"></td>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_dubins.gif" width="440" alt="HJ-Gauss Dubins-vehicle comparison"></td>
  </tr>
  <tr>
    <th>Double integrator</th>
    <th>Murmurations</th>
  </tr>
  <tr>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_double_integrator.gif" width="440" alt="HJ-Gauss double-integrator comparison"></td>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_murmuration.gif" width="440" alt="HJ-Gauss murmuration comparison"></td>
  </tr>
</table>

### Simultaneous pairwise reachability filtering for lifelong MAPF

Paper: [Simultaneous Pairwise Reachability Filtering for Lifelong MAPF](https://scriptedonachip.com/downloads/Papers/icra27.pdf)

The [`levelsetpy/mapf/`](levelsetpy/mapf/) project studies continuous execution of lifelong multi-agent path-finding plans under vehicle dynamics and disturbance. A three-dimensional Hamilton--Jacobi backward-reachable tube supplies a numerical candidate barrier for each ordered robot pair. The controller intersects the resulting affine steering constraints and selects the turn rate closest to the nominal command, with a least-violation fallback when the intersection is empty.

Across 30 paired 14-robot warehouse episodes, mean contacts decrease from `52.47` under geometric execution to `1.07` under simultaneous filtering. The one-constraint ablation records `0.73` contacts, so the paired comparison does not establish an advantage for either barrier controller. On 25 solved continuous-time conflict-based search schedules, simultaneous filtering decreases mean contacts from `17.56` to `0.44` while increasing final goal distance from `0.98` to `7.50` cells. These experiments measure contact reduction and its tracking cost; numerical barrier feasibility at sampled states is not a continuous-time safety guarantee.

```sh
cd levelsetpy/mapf
OPENBLAS_NUM_THREADS=1 pytest -q
python3 experiments/compatible_control_demo.py --plot
```

See the [Cole--Hopf README](levelsetpy/cole-hopf/README.md) for deterministic validation and rendering commands, the [HJ-Gauss README](levelsetpy/hjgauss/README.md) for the Monte Carlo evaluator and scaling study, and the [MAPF README](levelsetpy/mapf/README.md) for controller definitions, external baselines, reproduction commands, and limitations.

## Installation and prerequisites

### Dependencies

| :--: | :--: | :--: | :--: | :--: | :--: |
| [Numpy](https://numpy.org/) | [Scipy](https://scipy.org/) | [PyTorch](https://pytorch.org/) | [Absl-py](https://abseil.io/docs/python/quickstart) | [Scikit-image](https://scikit-image.org/) | [Matplotlib](https://matplotlib.org/) |
| :--: | :--: | :--: | :--: | :--: | :--: |

### Build and install

Activate your `conda` or `virtualven` environment, then install as follows

```bash
python setup.py build --build-lib=/path/to/your-desired/build
pip install -e . 
```


A separate `README.md` file is left in the respective folders.

## Examples

### Initial value problems (IVP)

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

### Robustly Controlled Backward Reachable Tubes (RCBRTs)



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



## Citing this work

The entries below follow the titles, authors, venues, and manuscript links on the [publications page](https://scriptedonachip.com/pubs).

```bibtex
@article{molu2025levelsetpy,
  title = {LevelSetPy: A GPU-Accelerated Package for Hyperbolic Hamilton--Jacobi Partial Differential Equations},
  author = {Molu, Lekan},
  journal = {ACM Transactions on Mathematical Software},
  year = {2025},
  doi = {10.1145/3730407},
  url = {https://scriptedonachip.com/downloads/Papers/levelsetpy.pdf}
}

@inproceedings{molu2024python,
  title = {The Python LevelSet Toolbox (LevelSetPy)},
  author = {Molu, Lekan},
  booktitle = {2024 IEEE 63rd Conference on Decision and Control (CDC)},
  pages = {8938--8945},
  year = {2024},
  organization = {IEEE},
  url = {https://scriptedonachip.com/downloads/Papers/levelsetpy_cdc.pdf}
}

@unpublished{molu2026colehopf,
  title = {Cole--Hopf Transformed Solvers for Viscous Hamilton--Jacobi Reachability},
  author = {Molu, Lekan},
  note = {Submitted to the Proceedings of the National Academy of Sciences},
  year = {2026},
  url = {https://scriptedonachip.com/downloads/Papers/colehopf.pdf}
}

@unpublished{molu2026samplinghj,
  title = {Sampling-based HJ Reachability},
  author = {Molu, Lekan and Renganathan, Venkatraman and Cho, Namhoon},
  note = {Submitted to Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences},
  year = {2026},
  url = {https://scriptedonachip.com/downloads/Papers/roysoc.pdf}
}

@unpublished{molu2026mapf,
  title = {Simultaneous Pairwise Reachability Filtering for Lifelong MAPF},
  author = {Molu, Lekan},
  note = {Submitted to the 2027 IEEE International Conference on Robotics and Automation},
  year = {2026},
  url = {https://scriptedonachip.com/downloads/Papers/icra27.pdf}
}
```
