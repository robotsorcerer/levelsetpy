### LevelSets Py

This repo is a reproduction of Ian Mitchell's LevelSets Toolbox in Python written by Lekan Molu in August, 2021

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

+ Fix len of smooth in derivL computation in upwindFirstWENO5.py
+ data_proj is a lotta issue
+ hyperpts in initialConditions
+ SpatialDerivative/Other Folder


Be sure your ordering is consistent for the following NUMPY classes/methods
# array, asarray, zeros, ones, diag, reshape, flatten,
