### Introduction

The Toolbox of Level Set Methods (ToolboxLS) is a collection of Matlab routines that implement numerical algorithms to approximate the solution of the time-dependent Hamilton-Jacobi (HJ) partial
differential equation (PDE) in any number of dimensions.  The HJ PDE is often used for simulating dynamic implicit surfaces in animation and computational fluid dynamics (CFD), and it is of independent
interest in areas of optimal control, differential games, robotics, financial mathematics, continuous reachability, etc.

### Backward Compatibility
To use ToolboxLS, you will need to run the basic version of Matlab. No additional toolboxes are required.  Version 1.1 of ToolboxLS definitely works with Matlab R2010 through R2023a, but likely works for some older
releases. 

If you got these files from the ToolboxLS web page in zip archive, you can find the most recent version of ToolboxLS (including bug fixes) at this Mercurial repository:

    https://bitbucket.org/ian_mitchell/toolboxls

If you are downloading from this repository, you can choose between the tagged versions or the tip.  Normally we suggest choosing the most recent tagged version -- it is most likely to be stable, but it may be
missing some bug fixes.

### Usage 

 If you want to run on very old releases of Matlab, you may need to remove certain recent language features by hand (eg: the ~parameter, && short-circuit boolean operator). Before using ToolboxLS, you will need to edit one configuration file as follows.  Go into the Examples/ subdirectory.  There you will find a file:

    addPathToKernel.m  

Open that file with your favorite text editor and modify the path name contained there so that it contains the absolute path (starting from root) of the Kernel subdirectory of this distribution.  Save the modified file.

Once you have edited this file, you can try out ToolboxLS as follows. 

+ Start Matlab, change to the Examples/Basic directory, and try running:

	convectionDemo.m

If it generates a figure showing a circle moving to the right, you are
on your way.

For more information and to get the 140 page, indexed user guide see

  http://www.cs.ubc.ca/~mitchell/ToolboxLS

Ian M. Mitchell
August 2012

Lekan Molu
April 2023