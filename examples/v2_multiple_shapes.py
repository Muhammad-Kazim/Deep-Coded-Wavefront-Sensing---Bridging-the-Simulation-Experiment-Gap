import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
from coded_wfs_sim import visualization
import numpy as np
import matplotlib.pyplot as plt


# Grid and propagation parameters setup
wl = 640e-9
spatial_resolution = [100e-9, 100e-9, 100e-9] # dx, dy, dz
grid_shape = [500, 500, 500] # x=0->, y=0->, z=0->
n_background = 1. # immersion medium RI

# Create the Geometry object with a shared grid
geometry = geometry.Geometry(grid_shape, spatial_resolution, n_background)

# Add shapes to the same grid
geometry.add_sphere(center=(40e-6, 40e-6, 20e-6), radius=5e-6, RI=1.5)
geometry.add_sphere(center=(10e-6, 10e-6, 20e-6), radius=5e-6, RI=1.5)
geometry.add_cube(center=(25e-6, 25e-6, 20e-6), side_length=5e-6, RI=1.5)

# Retrieve 3d RI distribution
RI_distribution = geometry.get_grid()

# Initial light field (Gaussian beam)
x = np.linspace(0, grid_shape[0]*spatial_resolution[0], grid_shape[0])
y = np.linspace(0, grid_shape[1]*spatial_resolution[1], grid_shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')
field = np.exp(-(X**2 + Y**2) / 0.1**2)

# Propagate and visualize
output_field = propagator.propagate_beam(field, RI_distribution, wl, spatial_resolution)
visualization.visualize_field(output_field)

print('=====================')
