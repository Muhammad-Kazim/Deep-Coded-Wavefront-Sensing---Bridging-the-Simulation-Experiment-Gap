import numpy as np
from scipy.fftpack import fft2, ifft2

# class
# propgate through medium from one plane to another
# propagte through geometry

def propagate_beam(field, refractive_index, wavelength, spatial_resolution):
    """
    Propagates the beam field using BPM.
    field: Input 2D complex field (x, y)
    refractive_index: Refractive index distribution
    wavelength: Wavelength of the light
    d: [dx, dy, Propagation step]
    """
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy, dz = spatial_resolution
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    k_perp2 = Kx**2 + Ky**2
    
    # Forward propagation
    for z in range(refractive_index.shape[2]):
        phase = np.exp(1j * k0 * (refractive_index[:, :, z] - 1) * dz)
        field = field * phase
        field_fft = np.fft2(field)
        transfer_function = np.exp(-1j * k_perp2 * dz / (2 * k0))
        field_fft = field_fft * transfer_function
        field = np.fft.ifft2(field_fft)
    return field



def propagate_beam_2(field, RI_distribution, RI_background, wavelength, spatial_resolution):
    """
    Propagates the beam field using BPM.
    field: Input 2D complex field (x, y)
    refractive_index: Refractive index distribution
    wavelength: Wavelength of the light
    d: [dx, dy, Propagation step]
    """
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy, dz = spatial_resolution
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    Kz = np.sqrt(0j + (k0*RI_background)**2 - Kx**2 - Ky**2)
    
    # Forward propagation
    for z in range(RI_distribution.shape[2]):
        field_fft = np.fft.fft2(field)
        transfer_function = np.exp(1j*Kz*dz)
        phase = np.exp(1j*k0*(RI_distribution[..., z] - RI_background)*dz)
        field = np.fft.ifft2(field_fft * transfer_function) * phase
        
    return field



def propagate(field, wavelength, spatial_resolution, dist):
    """Propagation through a homogenous medium

    Args:
        field (float): 2d complex field on a plane
        wavelength (float): if not air, than wl =/ RI_background
        spatial_resolution (): _description_
        dist (float): distance bw parallel planes in meters

    Returns:
        complex: field at parallel plane distance dist away 
    """
    
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy = spatial_resolution[:2]
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    Kz = np.sqrt(0j + k0**2 - Kx**2 - Ky**2)
    
    field_fft = np.fft.fft2(field)
    transfer_function = np.exp(1j*Kz*dist)
    field = np.fft.ifft2(field_fft * transfer_function)

    return field