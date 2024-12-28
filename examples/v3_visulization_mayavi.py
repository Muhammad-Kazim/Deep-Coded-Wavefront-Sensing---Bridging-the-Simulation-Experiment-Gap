import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
from coded_wfs_sim import visualization
import numpy as np


# Grid and propagation parameters setup
wl = 640e-9
spatial_resolution = [100e-9, 100e-9, 100e-9] # dx, dy, dz
grid_shape = [500, 500, 500] # x=0->, y=0->, z=0->
n_background = 1.33 # immersion medium RI
spatial_support = [spatial_resolution[i]*grid_shape[i] for i in range(3)]

# Create the Geometry object with a shared grid
geometry = geometry.Geometry(grid_shape, spatial_resolution, n_background)

# Add shapes to the same grid
for i in range(15):
    pos_x, pos_y, pos_z = np.random.randint(5, 45, size=3, dtype=int)
    geometry.add_cube(center=(pos_x*1e-6, pos_y*1e-6, pos_z*1e-6), side_length=5e-6, 
                      RI=np.random.randint(1, 5, size=1, dtype=int))

for i in range(15):
    pos_x, pos_y, pos_z = np.random.randint(5, 45, size=3, dtype=int)
    geometry.add_sphere(center=(pos_x*1e-6, pos_y*1e-6, pos_z*1e-6), radius=2.5e-6, 
                        RI=np.random.randint(1, 5, size=1, dtype=int))


# Retrieve 3d RI distribution
RI_distribution = geometry.get_grid()

# rendering slow but interactive visualization is very fast.
# visualization.visualize_grid_vol(RI_distribution, spatial_support, n_background=n_background, factor=2)
visualization.visualize_grid_vol(RI_distribution, n_background=n_background, factor=2)


# interactive visulization very slow for large arrays.
# possibly useful for small arrays (large factor).
visualization.visualize_grid(RI_distribution, n_background, factor=10)

# Initial light field
field = np.ones([grid_shape[0], grid_shape[1]])

# Propagate and visualize
output_field = propagator.propagate_beam(field, RI_distribution, wl, spatial_resolution)
visualization.visualize_field(output_field, spatial_support)

print('=====================')
