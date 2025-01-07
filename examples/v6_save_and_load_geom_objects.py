import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
from coded_wfs_sim import visualization
from coded_wfs_sim import utils
import numpy as np


### geometry objects saving and loading.

# Grid and propagation parameters setup
wl = 640e-9
spatial_resolution = [100e-9, 100e-9, 100e-9] # dx, dy, dz
grid_shape = [500, 500, 500] # x=0->, y=0->, z=0->
n_background = 1.51 # immersion medium RI
spatial_support = [spatial_resolution[i]*grid_shape[i] for i in range(3)]

# Create the Geometry object with a shared grid
geom = geometry.Geometry(grid_shape, spatial_resolution, n_background)

plane_pnt = [0, 0, 25e-6]
plane_normal = [0, 0, 1]

num = 100 # create num elemets
for i in range(num):
    print(f"Shapes: {i+1}/{num}", end="\r")
    pos_x, pos_y = np.random.randint(1, 49, size=2, dtype=int)
    pos_x, pos_y = pos_x*1e-6, pos_y*1e-6
    geom.add_obj_on_plane('cube', (pos_x, pos_y), length=1e-6, RI=1.4609, 
                              plane=[plane_pnt, plane_normal], bias=5e-6)

geom.add_plane(point=plane_pnt, normal=plane_normal, RI=1.49, thickness=10e-6)

# Retrieve 3d RI distribution
RI_distribution = geom.get_grid()
print('Geometry: Done')

# visualization
visualization.visualize_grid_vol(RI_distribution, n_background=n_background, factor=2)

# saving object
geom.save('examples/data/geometry/v1_test_geom.pkl')

del geom

# loading geom
geom2 = utils.load_pkl('examples/data/geometry/v1_test_geom.pkl')

# visualization
visualization.visualize_grid_vol(geom2.get_grid(), n_background=geom2.n_0, factor=2)

print(f"Loaded object: {geom2}")

print('=====================')
