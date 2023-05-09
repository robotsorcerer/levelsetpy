### Boundary Conditions 

At the heart of every (hyperbolic) partial differential equation with two or more independent variables are fixed-point boundary conditions. These conditions specify the values (or derivatives of the values) of the solution to the PDE at the edge of the computational domain. In this submodule, we consider [Dirichlet boundary conditions](../BoundaryCondition/add_ghost_dirichlet.py), [Neumann boundary condition](../BoundaryCondition/add_ghost_neumann.py), [periodic boundary conditions](../BoundaryCondition/add_ghost_periodic.py), and [extrapolated periodic conditions](../BoundaryCondition/add_ghost_extrapolate.py). A convenience function is included in that adds the same type of specified boundary condition in the `grid` data structure along every dimension of a computational domain. Since we implement the solutions to the PDEs of interest in this library on a fixed domain, we have implemented various boundary conditions to wit:

#### [Dirichlet boundary condition](../BoundaryCondition/add_ghost_dirichlet.py)

The Dirichlet boundary condition is a first-type boundary condition which when imposed on an ordinary or partial differential equation specifies the values of the solution along a boundary of the domain that defines the partial differential equation.

An example of a partial differential equation with Dirichlet boundary condition would be 

$$ u_t(x,t) &= \epsilon u_{xx}(x,t) + \lambda(x) u(x,t) + g(x) u_x(0, t) + \int_0^x f(x,y)u(y, t) dy, \\ 
u(0, t) &= 0 \\
u(1, t) &= \pi 
$$

defined over the triangular domain $\mathcal{T} = \{0 \le x \le 1}$ for all $t>0$. We say the boundary conditions $u(0, t)$ and $u(1, t)$ are of the Dirichlet type since they define the values of the solution to the PDE at the boundaries of the computational domain.

The calling signature is 

```python 
    addGhostDirichlet(dataIn, dim, width=None, ghostData=None)
```

where `dataIn` is the input data used to populate boundary cells within the computational domain; `dim` is the dimension of the Cartesian grid along which we intend to populate the boundary data; `width` is the size of the stencil used for manipulating data across domain boundaries; and `ghostData` contains the lower and upper fixed points of the boundary conditions. 

##### Implementation details

Our implementation of this is straightforward assuming the solution to the differential equation is defined on a Cartesian grid: along the Cartesian dimension where the boundary to the differential equation is to be populated (this is usually the 3rd dimension), we proceed as follows: 

+ We first create an output data array (call this `dataOut`) with the same shape as the user-specified input data (`dataIn`). Here, `dataIn` represents the solution of the PDE within the domain and at the fixed boundary nodes for the domain of interest.

+ Along the dimension of the shape of `dataOut` where we want to insert the boundary conditions, we pad the dimension with an extra node (with values zero) on either side of the dimension. This is our default; the `width` of the stencil can vary based on user-specifications.
    
+ What follows is a systematic copying over of `dataIn` into `dataOut` excluding the first `2*width` nodes along the dimenson that corresponds to the boundary. 
    
+ Seeing we are dealing with a Dirichlet boundary condition, for every extra node in the boundary data where the stencil points, the _solution of the differential equation_ is then added on the grid nodes corresponding to stencil indices. This solution is specified in the Bundle structure `ghostData` as either `lowerValue` or `upperValue`.

For further details, please consult the module [addGhostNeumann](../BoundaryCondition/add_ghost_neumann.py).


For further details, please consult the module [addGhostDirichlet](../BoundaryCondition/add_ghost_dirichlet.py).

#### [Neumann boundary condition](../BoundaryCondition/add_ghost_neumann.py)

The Neumann boundary condition is the classical second-type of boundary condition which when imposed on a partial differential equation, specifies the derivatives of the corresponding values of interest at the boundary of the computational domain. 

An example of a partial differential equation with Dirichlet boundary condition would be 

\begin{align} 
u_t(x,t) &= \epsilon u_{xx}(x,t) + \lambda(x) u(x,t) + g(x) u_x(0, t) + \int_0^x f(x,y)u(y, t) dy, \\ 
u_x(0, t) &= 0 \\
u_x(1, t) &= \pi 
\end{align}

defined over the triangular domain $\mathcal{T} = \{0 \le x \le 1}$ for all $t>0$. We say the boundary conditions $u(0, t)$ and $u(1, t)$ are of the Neumnann type since they define the derivatives of the solution to the PDE at the boundaries of the computational domain of the partial differential equation. 

The calling signature is 

```python 
    addGhostNeumann(dataIn, dim, width=None, ghostData=None)
```

where `dataIn` is the input data meant to populate boundary cells within the computational domain; `dim` is the dimension of the Cartesian grid along which we intend to populate the boundary data; `width` is the size of the stencil used for manipulating data across domain boundaries; and `ghostData` contains the lower and upper fixed points of the boundary conditions.

##### Implementation details
    
Our implementation of this is straightforward assuming the solution to the differential equation is defined on a Cartesian grid: along the Cartesian dimension where the boundary to the differential equation is to be populated (this is usually the 3rd dimension), we proceed as follows: 

+ We first create an output data array (call this `dataOut`) with the same shape as the user-specified input data (`dataIn`). Here, `dataIn` represents the solution of the PDE within the domain and at the fixed boundary nodes for the domain of interest.

+ Along the dimension of the shape of `dataOut` where we want to insert the boundary conditions, we pad the dimension with an extra node (with values zero) on either side of the dimension. This is our default; the `width` of the stencil can vary based on user-specifications.
    
+ What follows is a systematic copying over of `dataIn` into `dataOut` excluding the first `2*width` nodes along the dimenson that corresponds to the boundary. 
    
+ Seeing we are dealing with a Neumann boundary condition, for every extra node in the boundary data stencil parameter, the derivative to the solution of the differential equation is first computed via finite differences and is then padded on the grid nodes corresponding to stencil indices in `dataOut`.

For further details, please consult the module [addGhostNeumann](../BoundaryCondition/add_ghost_neumann.py).

#### [Extrapolated boundary condition](../BoundaryCondition/add_ghost_extrapolate.py)

This boundary condition linearly extrapolates data from the edge of a computational domain of interest, taking cognizance of the derivative (computed via finite differencing) of the solution to the PDE of interest along the boundaries. 

##### Implementation details
    
Our implementation of this is straightforward assuming the solution to the differential equation is defined on a Cartesian grid: along the Cartesian dimension where the boundary to the differential equation is to be populated (this is usually the 3rd dimension), we proceed as follows: 


+ We first create an output data array (call this `dataOut`) with the same shape as the user-specified input data (`dataIn`). Here, `dataIn` represents the solution of the PDE within the domain and at the fixed boundary nodes for the domain of interest.

+ Choosing the sign of the derivative such that our would-be extrapolation goes toward  or away from the zero levelset, we compute the derivatives from the data points on the top and bottom edges of a Cartesian grid dimension. Our scheme adopts a multiplicative factor of +1 if the derivatives go towards the zero levelset; or we choose a multiplicative factor of -1 if the derivatives go away from the zero levelset.


# ToDos
+ [Periodic boundary condition](../BoundaryCondition/add_ghost_periodic.py); and
+ [Repeated boundary condition](../BoundaryCondition/add_ghost_all.py)