import numpy as np

## create class such that object holds the RI distribution
## type check input, return
def create_sphere(grid_shape, spatial_resolution, center, radius, n_sphere, n_background):
    """
    Create a 3D refractive index distribution with a sphere.
    grid: Tuple of (Nx, Ny, Nz)
    center: Tuple of sphere center (cx, cy, cz)
    radius: Sphere radius
    n_sphere: Refractive index of the sphere
    n_background: Background refractive index
    """
    x = np.linspace(0, grid_shape[0]*spatial_resolution[0], grid_shape[0])
    y = np.linspace(0, grid_shape[1]*spatial_resolution[1], grid_shape[1])
    z = np.linspace(0, grid_shape[2]*spatial_resolution[2], grid_shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    refractive_index = np.full(grid_shape, n_background)
    refractive_index[distance <= radius] = n_sphere

    return refractive_index


