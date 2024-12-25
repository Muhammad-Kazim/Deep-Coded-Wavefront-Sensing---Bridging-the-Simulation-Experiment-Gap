import numpy as np
from scipy.fftpack import fft2, ifft2

# class
# propgate through medium from one plane to another
# propagte through geometry
def propagate_beam(field, refractive_index, wavelength, spatial_resoltion):
    """
    Propagates the beam field using BPM.
    field: Input 2D complex field (x, y)
    refractive_index: Refractive index distribution
    wavelength: Wavelength of the light
    d: [dx, dy, Propagation step]
    """
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy, dz = spatial_resoltion
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    k_perp2 = Kx**2 + Ky**2
    
    # Forward propagation
    for z in range(refractive_index.shape[2]):
        phase = np.exp(1j * k0 * (refractive_index[:, :, z] - 1) * dz)
        field = field * phase
        field_fft = fft2(field)
        transfer_function = np.exp(-1j * k_perp2 * dz / (2 * k0))
        field_fft = field_fft * transfer_function
        field = ifft2(field_fft)
    return field
