import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
from coded_wfs_sim import visualization
from coded_wfs_sim import utils
import numpy as np


### adding two geometry objects.

# loading geom
geom1 = utils.load_pkl('examples/data/geometry/v1_test_geom.pkl')
print(f"Loaded object 1: {geom1}")
visualization.visualize_grid_vol(geom1.get_grid(), n_background=geom1.n_0, factor=2)

geom2 = utils.load_pkl('examples/data/geometry/v2_test_geom.pkl')
print(f"Loaded object 2: {geom2}")
visualization.visualize_grid_vol(geom2.get_grid(), n_background=geom2.n_0, factor=2)

geom3 = geom1 + geom1
geom3.add_sphere([25e-6, 25e-6, 50e-6], 5e-6, 1.4)
print(f"Object 3: {geom3}")
visualization.visualize_grid_vol(geom3.get_grid()[:, :, ::2], n_background=geom3.n_0, factor=2)

# outputs error
geom3 = geom1 + geom2
print(geom3)

print('=====================')
