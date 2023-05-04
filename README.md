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
| [Absl-py](https://abseil.io/docs/python/quickstart)   | [Scikit-image](https://scikit-image.org/) | [Matplotlib](https://matplotlib.org/)     | 

**Note**: The user should elect to install a cupy version that matches the version of their CUDA installation. Cupy is commented out in the [requirements.txt](requirements.txt) file. Please follow the directions on the [cupy install page](https://docs.cupy.dev/en/stable/install.html).

### **Build and Install**

Be sure to activate your `conda` or `virtualven` environment first e.g. if your environment is named `py39`:

```bash
	conda activate py39
```

Then install to the activated environment as follows

+ BUILD: `python setup.py build --build-lib=/path/to/your-desired/build`

+ INSTALL: via pip: `pip install -e . `


A separate `README.md` file is left in the respective folders for each package.


### Citing this work

If you have found this library of routines and packages useful, please cite it:

```
@article{LevPy,
title   = {LevelSetPy: A GPU-Accelerated Python Software Package for Resolving Hamilton-Jacobi PDEs and Level Set Evolutions},
author  = {Molu, Lekan},
howpublished = {\url{https://github.com/robotsorcerer/LevelSetPy}},
note = {Accessed March 31, 2023}
year    = {2023},
}
```
