__all__ = ["implicit_mesh"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from LevelSetPy.Utilities import Bundle

def implicit_mesh(surface, level, spacing, gd='ascent', edge_color='k', face_color='r'):

    """
        Generate a level set about the avg of the min and max of the vol
        of an implicit surface function. This algorithm uses the Marching Cubes Function of Lewiner et al.
        For this implementation, we are leveraging the Lewiner's method implemeneted in scipykit.image measure
        Lorensen et al's algorithm:   Lorensen, W. E.; Cline, Harvey E. (1987).
        "Marching cubes: A high resolution 3d surface construction algorithm".
        ACM Computer Graphics. 21 (4): 163–169. CiteSeerX 10.1.1.545.613. doi:10.1145/37402.37422

        Inputs:
            surface: A signed-distance representation of the implicit surface or a 3D surface volume

            level: Contour value to search for isosurfaces in `volume`. If not
            given or None, the average of the min and max of vol is used.

            spacing : length-3 tuple of floats Voxel spacing in spatial dimensions corresponding to numpy array
            indexing dimensions (M, N, P) as in `volume`.

            gd: Gradient_direction; Controls if the mesh was generated from an isosurface with gradient
                descent toward objects of interest (the default), or the opposite,
                considering the *left-hand* rule.

            edge_color: color of the edge of the mesh.

            face_color: color of the face of the mesh.
        Returns
        =======
            3D Mesh similar to the one you get in matlab

        Author: Lekan Molu, September 07, 2021
    """
    # Use marching cubes to obtain the surface mesh of the implicit function
    verts, faces, normals, values = measure.marching_cubes(surface, level=level, spacing=spacing, gradient_direction=gd)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)

    
    # we now return faces in order to set the limits of the plot in pyplot
    return Bundle(dict(mesh=mesh, verts=verts))