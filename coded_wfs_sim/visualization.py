import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab


# https://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
def visualize_grid_vol(RI_distribution, support=None, n_background=1., factor=2, 
                       title="3D RI Distribution"):
    """3d visualization of the gird.

    Args:
        RI_distribution (3d array): 3d space with RI at each point.
        n_background (float, optional): background RI. Defaults to 1.
        factor (int, optional): downsampling, use every factor-th sample. Defaults to 2.
    """
    x = np.arange(0, RI_distribution.shape[0], factor)
    y = np.arange(0, RI_distribution.shape[1], factor)
    z = np.arange(0, RI_distribution.shape[2], factor)
    
    # Axis extent in um
    # Plots differently (blurry). Not sure why.
    if support is not None:
        x = x*support[0]/RI_distribution.shape[0]/1e-6
        y = y*support[1]/RI_distribution.shape[1]/1e-6
        z = z*support[2]/RI_distribution.shape[2]/1e-6

    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing="ij")
    RI_abs = np.abs(RI_distribution - n_background)
        
    # Plot scatter with mayavi
    figure = mlab.figure(title)
    
    grid = mlab.pipeline.scalar_field(x_mesh, y_mesh, z_mesh, 
                                      RI_abs[::factor, ::factor, ::factor])

    mlab.pipeline.volume(grid, vmin=RI_abs.min(), vmax=RI_abs.max())
    
    mlab.axes()
    mlab.show()
    

# https://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
def visualize_grid_iso_surf(RI_distribution, n_background=1., factor=2, 
                            title='3D Iso-surfaces Visualization'):
    """3d visualization of iso-surfaces.

    Args:
        RI_distribution (3d array): 3d space with RI at each point.
        n_background (float, optional): background RI. Defaults to 1.
        factor (int, optional): downsampling, use every factor-th sample. Defaults to 2.
    """
    x = np.arange(RI_distribution.shape[0]//factor)
    y = np.arange(RI_distribution.shape[1]//factor)
    z = np.arange(RI_distribution.shape[2]//factor)

    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing="ij")
    RI_abs = np.abs(RI_distribution - n_background)
    # Plot scatter with mayavi
    figure = mlab.figure(title)
    
    mlab.contour3d(x_mesh, y_mesh, z_mesh, 
                                      RI_abs[::factor, ::factor, ::factor])
    mlab.axes()
    mlab.show()


def visualize_grid(RI_distribution, n_background=1., factor=10):
    """3d-grid visualization. Will kill program for large grid.

    Args:
        RI_distribution (float): 3d space with RI at each point.
        n_background (float): background RI.
        factor (int): downsampling, use every factor-th sample.  
    """
    x = np.arange(RI_distribution.shape[0]//factor)[:, None, None]
    y = np.arange(RI_distribution.shape[1]//factor)[None, :, None]
    z = np.arange(RI_distribution.shape[2]//factor)[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, s=(np.abs(RI_distribution-n_background))[::factor, ::factor, ::factor]*factor*10, 
               alpha=0.25)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.show()

def visualize_field(field, support, title="Field Intensity", units=1e-6):
    plt.imshow(np.abs(field)**2, cmap='inferno', extent=[0, support[0]/units, 0, support[1]/units])
    plt.colorbar()
    plt.title(title)
    if units == 1e-6:
        plt.xlabel('X (um)')
        plt.ylabel('Y (um)')
    elif units == 1e-3:
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
    else:
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
    plt.show()