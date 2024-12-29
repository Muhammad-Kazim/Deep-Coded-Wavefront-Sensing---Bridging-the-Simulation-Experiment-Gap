import numpy as np

# add cuboids, hemisphere, prisms
# handle intersecting objects
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

        # Compute distance from the center
        distance = np.sqrt(
            (self.x_mesh - cx)**2 + 
            (self.y_mesh - cy)**2 + 
            (self.z_mesh - cz)**2
        )
        
        # Logical mask for the sphere
        sphere_mask = distance <= radius
        self.grid[sphere_mask] = RI
        
    def add_plane(self, point, normal, RI, thickness=None):
        """
        Add a thick plane to the grid. Physical coordinates.

        Args:
            point (float): point that lies on the plane.
            normal (float): normal to the planes.
            RI (float): RI of plane.
            thickness (float, optional): Thickness/2 on either halfspace.
        """
        px, py, pz = point
        nx, ny, nz = normal
        
        if thickness is None:
            thickness = 2*self.dz
            
        # Plane equation: n . (x - p) = 0
        mask = np.abs(nx * (self.x_mesh - px) + 
                      ny * (self.y_mesh - py) + 
                      nz * (self.z_mesh - pz)) <= thickness / 2
        
        self.grid[mask] = RI
        
    def add_obj_on_plane(self, object, center, length, RI, 
                         plane, bias=0.):
        """Draw shapes along a plane. Does not draw plane.
        Adds shapes with centers along the plane, however, the cubes
        are not parallel to the plane but to the XYZ axis. Maybe, 
        rotation in the end required. Probably not problematic for small
        shapes.

        Args:
            object (string): specify shape from available shapes ("cube"/"spehre").
            center (float): center of shape in XY plane. Z is calculated.
            length (float): side_length/radius, depeding on shape.
            RI (float): refractive index of shape.
            plane (list[float]): [[point], [normal]]
            bias (float): distance along -Z from center.
        """
        
        plane_pnt = plane[0]
        plane_normal = plane[1]

        cnt_x, cnt_y = center
        cnt_z = -1*(plane_normal[0]*(cnt_x - plane_pnt[0]) + 
                plane_normal[1]*(cnt_y - plane_pnt[1]) - 
                plane_normal[2]*plane_pnt[2])/plane_normal[2]
        cnt_z -= bias
        
        if object == 'cube':
            self.add_cube(center=(cnt_x, cnt_y, cnt_z), side_length=length, RI=RI)
        elif object == 'sphere':
            self.add_sphere(center=(cnt_x, cnt_y, cnt_z), radius=length, RI=RI)
        else:
            raise TypeError(f'Object {object} not available.')

    def get_grid(self):
        """
        Return the current grid with all shapes added.
        """
        return self.grid
