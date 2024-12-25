import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim.geometry import create_sphere
from coded_wfs_sim.propagator import propagate_beam
from coded_wfs_sim.visualization import visualize_field
import numpy as np
import matplotlib.pyplot as plt


# Grid and propagation parameters setup
wl = 640e-9
spatial_resolution = [100e-9, 100e-9, 100e-9] # dx, dy, dz
grid_shape = [256, 256, 256] # x, y, z=0->

print(f'''Coordiante system with size: \n 
      X = [0, {spatial_resolution[0]*grid_shape[0]:.2e}], Res_X = {spatial_resolution[0]}
      Y = [0, {spatial_resolution[1]*grid_shape[1]:.2e}], Res_Y = {spatial_resolution[1]}
      Z = [0, {spatial_resolution[2]*grid_shape[2]:.2e}], Res_Z = {spatial_resolution[2]}
      ''')

# Add geometry
refractive_index = create_sphere(
    grid_shape, spatial_resolution, center=(12.5e-6, 12.5e-6, 12.5e-6), radius=5e-6, 
    n_sphere=1.5, n_background=1.0
    )

# Initial light field (Gaussian beam)
x = np.linspace(0, grid_shape[0]*spatial_resolution[0], grid_shape[0])
y = np.linspace(0, grid_shape[1]*spatial_resolution[1], grid_shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')
field = np.exp(-(X**2 + Y**2) / 0.1**2)

# Propagate and visualize
output_field = propagate_beam(field, refractive_index, wl, spatial_resolution)
visualize_field(output_field)
