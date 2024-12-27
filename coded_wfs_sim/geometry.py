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


import numpy as np

class Geometry:
    def __init__(self, grid_shape, spatial_resolution, n_background):
        """
        Initialize a single grid with background refractive index.
        
        grid_shape: Tuple of (nx, ny, nz) defining the grid dimensions.
        spatial_resolution: Tuple of (dx, dy, dz) defining spatial resolution.
        n_background: Background refractive index n_0.
        """
        
        print(f'''Coordiante system with size: \n 
              X = [0, {spatial_resolution[0]*grid_shape[0]:.2e}], Res_X = {spatial_resolution[0]}
              Y = [0, {spatial_resolution[1]*grid_shape[1]:.2e}], Res_Y = {spatial_resolution[1]}
              Z = [0, {spatial_resolution[2]*grid_shape[2]:.2e}], Res_Z = {spatial_resolution[2]}
              Immersion RI: {n_background}
      ''')
        
        self.dx, self.dy, self.dz = spatial_resolution
        self.nx, self.ny, self.nz = grid_shape
        self.n_0 = n_background

        # Initialize the grid and meshgrid
        self.grid = np.ones([self.nx, self.ny, self.nz])*self.n_0

        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        z = np.arange(self.nz) * self.dz
        self.x_mesh, self.y_mesh, self.z_mesh = np.meshgrid(x, y, z, indexing="ij")

    def add_cube(self, center, side_length, RI):
        """
        Add a cube to the grid.
        
        center: Tuple of (cx, cy, cz) defining the cube's center in real units.
        side_length: Length of the cube's side in real units.
        RI: refractive index of homogenous shape.
        """
        cx, cy, cz = center
        s = side_length / 2

        # Logical masks for the cube
        cube_mask = (
            (self.x_mesh >= cx - s) & (self.x_mesh <= cx + s) &
            (self.y_mesh >= cy - s) & (self.y_mesh <= cy + s) &
            (self.z_mesh >= cz - s) & (self.z_mesh <= cz + s)
        )
        self.grid[cube_mask] = RI

    def add_sphere(self, center, radius, RI):
        """
        Add a sphere to the grid.
        
        center: Tuple of (cx, cy, cz) defining the sphere's center in real units.
        radius: Radius of the sphere in real units.
        RI: refractive index of homogenous shape.
        """
        cx, cy, cz = center

        # Compute squared distance from the center
        distance = np.sqrt(
            (self.x_mesh - cx)**2 + 
            (self.y_mesh - cy)**2 + 
            (self.z_mesh - cz)**2
        )
        
        # Logical mask for the sphere
        sphere_mask = distance <= radius
        self.grid[sphere_mask] = RI

    def get_grid(self):
        """
        Return the current grid with all shapes added.
        """
        return self.grid
