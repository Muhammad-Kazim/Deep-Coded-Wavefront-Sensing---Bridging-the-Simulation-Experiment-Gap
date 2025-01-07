import sys
sys.path.insert(0, "/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
from coded_wfs_sim import visualization
import numpy as np
import scipy
from tifffile import tifffile


## reference speckle pattern

load_grid = scipy.io.loadmat('v1_diffuser.mat')

# Grid and propagation parameters setup
diffuser_RI_distribution = load_grid['RI_distribution']
wl = load_grid['wavelength']
spatial_resolution = list(load_grid['spatial_resolution'][0]) # dx, dy, dz
grid_shape = diffuser_RI_distribution.shape # x=0->, y=0->, z=0->
n_background = load_grid['RI_background'] # immersion medium RI
plane_position_z = list(load_grid['plane_point'][0])[-1]


spatial_support = [spatial_resolution[i]*grid_shape[i] for i in range(3)]

diffuser_RI_distribution = diffuser_RI_distribution[:, :, 150:300]
spatial_resolution[-1] = spatial_resolution[-1]

# visualization
visualization.visualize_grid_vol(diffuser_RI_distribution, n_background=n_background, factor=2)

# Initial light field
field = np.ones([grid_shape[0], grid_shape[1]])*100

# Propagate and visualize
ref_field = propagator.propagate_beam_2(field, diffuser_RI_distribution, n_background, wl, spatial_resolution)
tifffile.imwrite('ref_speckle.tif', np.abs(ref_field)**2)

visualization.visualize_field(ref_field, spatial_support)

print('=====================')


## bead speckle pattern


print('=====================')
