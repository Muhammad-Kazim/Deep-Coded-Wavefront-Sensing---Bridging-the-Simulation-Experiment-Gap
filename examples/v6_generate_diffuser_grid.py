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

geometry.add_sphere(center=(25e-6, 25e-6, 15e-6), radius=5e-6, RI=1.4609)

plane_pnt = [0, 0, 30e-6]
plane_normal = [0, 0, 1]

for i in range(500):
    print(f"Shapes: {i+1}/500", end="\r")
    pos_x, pos_y = np.random.randint(1, 49, size=2, dtype=int)
    pos_x, pos_y = pos_x*1e-6, pos_y*1e-6
    geometry.add_obj_on_plane('cube', (pos_x, pos_y), length=1e-6, RI=1.4609, 
                              plane=[plane_pnt, plane_normal], bias=5e-6)

geometry.add_plane(point=plane_pnt, normal=plane_normal, RI=1.49, thickness=10e-6)

# Retrieve 3d RI distribution
RI_distribution = geometry.get_grid()
print('Geometry: Done')

# visualization
visualization.visualize_grid_vol(RI_distribution, n_background=n_background, factor=2)

# Initial light field
field = np.ones([grid_shape[0], grid_shape[1]])

# Propagate and visualize
output_field = propagator.propagate_beam_2(field, RI_distribution, n_background, wl, spatial_resolution)
visualization.visualize_field(output_field, spatial_support)

print('=====================')
