__all__ = ["shapeUnion", "shapeIntersection", "shapeDifference",  "shapeComplement"]

__author__ 		  = "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	  = "There are None."
__license__ 	  = "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		  = "patlekno@icloud.com"
__status__ 		  = "Completed"


import numpy as np
import logging
logger = logging.getLogger(__name__)

def shapeUnion(shapes):
    """
     shapeUnion: implicit surface function for the union of two shapes.

       data = shapeUnion(shapes)

     Creates an implicit surface function for the union of two shapes
       which are themselves defined by implicit surface functions.

     The union is created by taking the pointwise minimum of the functions.

     See O&F, pg 10, Geometry Toolbox.

     Input:
       shapes (*.ndarray): Implicit surface data array as tuples for two or more interfaces.
     
     Output:
       data (*.ndarray): Implicit surface of the union (same size as one of shapes) of shapes.

     Lekan Molu, September, 2021
    """
    if len(shapes)==2:
      data = np.minimum(shapes[0], shapes[1])
    else:
      data = np.minimum.reduce(shapes)

    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data

def shapeIntersection(shapes):
    """
     shapeIntersection: implicit surface function for the intersection of two shapes.

       data = shapeIntersection(shapes)

     Creates an implicit surface function for the intersection of two or more shapes
       which are themselves defined by implicit surface functions.

     The intersection is created by taking the pointwise maximum of the functions.

      Creates an implicit surface function for the intersection of two shapes
        which are themselves defined by implicit surface functions.

      The intersection is created by taking the pointwise maximum of the functions.

      See O&F, pg 10, Geometry Toolbox.

     Input
     =====
       shapes (*.ndarray): Implicit surface data as tuples for two or more interfaces.
     
     Output
     ======
       data (*.ndarray): Implicit surface of the intersection (same size as one of shapes) of `shapes`.

      Lekan Molu, September, 2021
    """
    if len(shapes)==2:
      data = np.maximum(shapes[0], shapes[1])
    else:
      data = np.maximum.reduce(shapes)

    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data

def shapeDifference(shapes):
    """
     shapeUnion: implicit surface function for the difference of two shapes.

       data = shapeDifference(shapes)

     Creates an implicit surface function for the difference of two or more shapes
      individually described by implicit surface functions.

     The difference is created by intersecting the first shape with the complement of the 
     second shape (or other shapes in the list) by utilizing the pointwise maximum operation.

     If the shapes are defined by signed distance functions,
       the resulting difference function will be close to but not exactly
       a signed distance function.

      Input
      =====
       shapes (*.ndarray): Implicit surface data or a tuple for two or more interfaces.
     
     Output
     ======
       data (*.ndarray): Difference between shapes[0] and the rest of shapes[1:].

      Lekan Molu, September, 2021
    """
    
    if len(shapes)==2:
      data = np.maximum(shapes[0], -shapes[1])
    else:
      data = np.maximum(shapes[0], *[-1 *s for s in shapes[1:]])

    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data

def shapeComplement(shape):
    """
     shapeComplement: implicit surface function for the complement of a shape.

       data = shapeComplement(shape)

     Creates an implicit surface function for the complement of a shape, 
      defined by implicit surface functions.

     The complement is created by taking the pointwise negation of shape.


      Input
      =====
       shape (*.ndarray): Implicit surface data for the interface of concern.
     
     Output
     ======
       data (*.ndarray): Complement of shape.

      Lekan Molu, September, 2021
    """
    data = -shape

    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')

    return data
