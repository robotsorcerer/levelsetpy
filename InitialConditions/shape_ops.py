import numpy as np
import logging
from Utilities import FLAGS
logger = logging.getLogger(__name__)

def shapeUnion(shape1, shape2):
    """
     shapeUnion: implicit surface function for the union of two shapes.

       data = shapeUnion(shape1, shape2)

     Creates an implicit surface function for the union of two shapes
       which are themselves defined by implicit surface functions.

     The union is created by taking the pointwise minimum of the functions.

     See O&F, pg 10, Geometry Toolbox.

     parameters:
       shape1      Implicit surface function data array for one shape.
       shape2      Implicit surface function data array for the other shape.

       data	Output data array (same size as shape1 and shape2)
                     containing an implicit surface function of the union.

     Lekan Molu, September, 2021
    """

    #---------------------------------------------------------------------------
    data = np.minimum(shape1, shape2)

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data

def shapeIntersection(shape1, shape2):
    """
     shapeIntersection: implicit surface function for the intersection of two shapes.

       data = shapeIntersection(shape1, shape2)

     Creates an implicit surface function for the intersection of two shapes
       which are themselves defined by implicit surface functions.

     The intersection is created by taking the pointwise minimum of the functions.

      Creates an implicit surface function for the union of two shapes
        which are themselves defined by implicit surface functions.

      The union is created by taking the pointwise minimum of the functions.

      See O&F, pg 10, Geometry Toolbox.

      parameters:
        shape1      Implicit surface function data array for one shape.
        shape2      Implicit surface function data array for the other shape.

        data	Output data array (same size as shape1 and shape2)
                      containing an implicit surface function of the union.

      Lekan Molu, September, 2021
    """

    #---------------------------------------------------------------------------
    data = np.maximum(shape1, shape2, order=FLAGS.order_type)

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten(order=FLAGS.order_type) < 0) or (np.all(data.flatten(order=FLAGS.order_type) > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data

def shapeDifference(shape1, shape2):
    """
     shapeUnion: implicit surface function for the difference of two shapes.

       data = shapeDifference(shape1, shape2)

     Creates an implicit surface function for the difference of two shapes
       which are themselves defined by implicit surface functions.

     The difference is created by taking the pointwise minimum of the functions.

     If the two shapes are defined by signed distance functions,
       the resulting difference function will be close to but not exactly
       a signed distance function.

     parameters:
       shape1      Implicit surface function data array for one shape.
       shape2      Implicit surface function data array for the other shape.

       data	Output data array (same size as shape1 and shape2)
                     containing an implicit surface function of the difference.

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 6/23/04
    """

    #---------------------------------------------------------------------------
    data = np.maximum(shape1, -shape2, order=FLAGS.order_type)

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten(order=FLAGS.order_type) < 0) or (np.all(data.flatten(order=FLAGS.order_type) > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data

def shapeComplement(shape):
    """
     shapeUnion: implicit surface function for the complement of two shapes.

       data = shapeComplement(shape)

     Creates an implicit surface function for the complement of two shapes
       which are themselves defined by implicit surface functions.

     The complement is created by taking the pointwise minimum of the functions.

     If the two shapes are defined by signed distance functions,
       the resulting complement function will be close to but not exactly
       a signed distance function.

     parameters:
       shape1      Implicit surface function data array for one shape.
       shape2      Implicit surface function data array for the other shape.

       data	Output data array (same size as shape1 and shape2)
                     containing an implicit surface function of the complement.

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 6/23/04
    """

    #---------------------------------------------------------------------------
    data = -shape

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten(order=FLAGS.order_type) < 0) or (np.all(data.flatten(order=FLAGS.order_type) > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data
