import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
from coded_wfs_sim import visualization
from coded_wfs_sim import utils

import numpy as np
from tifffile import tifffile
from matplotlib import pyplot as plt


### coded wfs sensing data creation.

# reference
geom1 = utils.load_pkl('examples/data/geometry/v4_diffuser_geom.pkl')
print(f"Loaded object 1: {geom1}")

RI_dist = geom1.get_grid()[:, :, :-400]
nx, ny, nz = geom1.nx, geom1.ny, geom1.nz
dx, dy, dz = geom1.dx, geom1.dy, geom1.dz
n_background = geom1.n_0
wl =  530e-9
NA = 1.15
M = 10
n_m = 1.

visualization.visualize_grid_vol(RI_dist, n_background=n_background, factor=2)

# Initial light field
field = np.ones([nx, ny])*80

# Propagate and visualize
output_field = propagator.propagate_beam_2(field, RI_dist, n_background, wl, [dx, dy, dz])
# output_field = propagator.propagate(output_field, wl, [dx, dy, dz], 1e-4)

visualization.visualize_field(output_field, [dx*nx, dy*ny])

tifffile.imwrite('examples/data/speckles/v6_ref.tif', (np.abs(output_field)**2).astype(np.uint16))

# object
geom2 = geometry.Geometry([nx, ny, 500], [dx, dy, dz], geom1.n_0)
geom2.add_sphere(center=(12.5e-6, 12.5e-6, 12.5e-6), radius=5e-6, RI=geom1.n_0+0.01)
# # visualization.visualize_grid_vol(geom2.get_grid(), n_background=geom2.n_0, factor=2)

# geom3 = geom2 + geom1
# visualization.visualize_grid_vol(geom3.get_grid(), n_background=n_background, factor=4)

# # Propagate and visualize
# # print(f'1, {field.dtype}')
output_field = propagator.propagate_beam_2(field, geom2.get_grid(), geom1.n_0, wl, [dx, dy, dz])
output_field = propagator.propagate(output_field, wl, [dx, dy, dz], 500*dz/2)

visualization.visualize_field(output_field, [dx*nx, dy*ny])
output_field = propagator.propagate_beam_2(output_field, RI_dist, n_background, wl, [dx, dy, dz])

# # print(f'2, {output_field.dtype}')
# output_field = propagator.propagate(output_field, wl/1.4, [dx, dy, dz], 125*50e-9)
visualization.visualize_field(output_field, [dx*nx, dy*ny])

# plt.imshow(np.angle(output_field))
# plt.colorbar()
# plt.show()

# print(f'3, {output_field.dtype}')
# output_field = propagator.propagate_beam_2(output_field, geom1.get_grid(), geom1.n_0, wl, [dx, dy, dz])
# visualization.visualize_field(output_field, [dx*nx, dy*ny])
# why such a bigh bright circle when radius is only 1 um?

# print(f'4, {output_field.dtype}')
# output_field = propagator.propagate(output_field, wl, [dx, dy, dz], 1e-3/20)

# x = np.arange(nx)*dx
# y = np.arange(ny)*dy
# x_mesh, y_mesh, = np.meshgrid(x, y, indexing="ij")
# alpha, beta = np.pi/3600, 0

# field = 80*np.exp(1j*(2*np.pi/wl)*(np.sin(alpha)*x_mesh + np.sin(beta)*y_mesh))

# plt.imshow(np.angle(field))
# plt.colorbar()
# plt.show()
# output_field = propagator.propagate_beam_2(field, geom1.get_grid(), geom1.n_0, wl, [dx, dy, dz])
# output_field = propagator.propagate(output_field, wl, [dx, dy, dz], 1e-4)

# visualization.visualize_field(output_field, [dx*nx, dy*ny])

# normalizing separately is probably wrong
# use max of object field propbably
# tifffile.imwrite('examples/data/speckles/v1_obj.tif', utils.normalization(np.abs(output_field)))
tifffile.imwrite('examples/data/speckles/v6_obj.tif', (np.abs(output_field)**2).astype(np.uint16))

# output_field = propagator.propagate_beam_2(field, geom3.get_grid(), geom3.n_0, wl, [dx, dy, dz])
# visualization.visualize_field(output_field, [dx*nx, dy*ny])
# tifffile.imwrite('examples/data/speckles/v1_obj_2.tif', utils.normalization(np.abs(output_field)))


# grains must be samller? reduce size? does number of elements have any effect?
print('=====================')
