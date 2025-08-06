import numpy as np
from matplotlib import pyplot as plt

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



def propagate_beam_2(field, RI_distribution, RI_background, wavelength, spatial_resolution, padding=None):
    """
    Propagates the beam field using BPM.
    field: Input 2D complex field (x, y)
    refractive_index: Refractive index distribution
    wavelength: Wavelength of light
    d: [dx, dy, Propagation step]
    """
    if padding:
        field = np.pad(field, padding, 'edge') # edge make sense
        # symmteric would add diffraction pattern from outside FOV
        # zeros will make the aperture diffraction pattern dominates the diffraction patter after a certain distance
        
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy, dz = spatial_resolution
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    Kz = np.sqrt(0j + (k0*RI_background)**2 - Kx**2 - Ky**2)
    
    transfer_function = np.exp(1j*Kz*dz)

    # Forward propagation
    for z in range(RI_distribution.shape[2]):
        field_fft = np.fft.fft2(field)
        # plt.imshow(np.abs(field_fft))
        # plt.show()
        transfer_function = np.exp(1j*Kz*dz)
        phase = np.exp(1j*k0*(RI_distribution[..., z] - RI_background)*dz)
        if padding:
            phase = np.pad(phase, padding, 'edge') # no delay in the padded region
            
        field = np.fft.ifft2(field_fft * transfer_function) * phase
    
    if padding:
        field = field[padding:-1*padding, padding:-1*padding]
    # print(field.dtype)
    return field


def propagate(field, wavelength, spatial_resolution, dist, padding=None, direction='forward', bandlimited=False):
    """Propagation through a homogenous medium

    Args:
        field (float): 2d complex field on a plane
        wavelength (float): if not air, than wl =/ RI_background
        spatial_resolution (): _description_
        dist (float): distance bw parallel planes in meters

    Returns:
        complex: field at parallel plane distance dist away 
    """
    if padding:
        field = np.pad(field, padding, 'edge') # edge make sense
        # symmteric would add diffraction pattern from outside FOV
        # zeros will make the aperture diffraction pattern dominates the diffraction patter after a certain distance
         
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy = spatial_resolution[:2]
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    Kz = 0j + k0**2 - Kx**2 - Ky**2
    if not np.all(Kz > 0):
        Kz[Kz < 0] = 0

    Kz = np.sqrt(Kz)
    
    field_fft = np.fft.fft2(field)
    if direction == 'backward':
        transfer_function = np.conj(np.exp(1j*Kz*dist))
    else:
        transfer_function = np.exp(1j*Kz*dist)
    
    if bandlimited:
        # max freq to prevent aliasing by the transfer function
        delU, delV = 1/(Nx*dx)*2*np.pi, 1/(Ny*dy)*2*np.pi
        uLimit = 1/(np.sqrt((2*delU*dist)**2 + 1)* wavelength)
        vLimit = 1/(np.sqrt((2*delV*dist)**2 + 1)* wavelength)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask = np.ones_like(transfer_function)
        mask[np.logical_or(np.abs(Kx) > int(uLimit), np.abs(Ky) >= int(vLimit))] = 0

        transfer_function = mask*transfer_function 
        
    field = np.fft.ifft2(field_fft * transfer_function)
    
    if padding:
        field = field[padding:-1*padding, padding:-1*padding]

    # print(field.dtype)
    return field


class Wave2d:
    """
    Given a wave components on a plane and optical parameters,
    calculates the wave components at another plane.
    """

    def __init__(self, numPx: list = [1392, 1040], 
                 sizePx: list = [0.00645, 0.00645], wl: float = 658*1e-6):
        
        ## setup params
        # wavelength, camera specs, resolutions, and limits

        self.mm = 1e-3 # standard otherwise specified
        ## Inputs
        self.wl = wl*self.mm

        ## Camera-based calculations or planes size and resolution
        self.numPx = numPx # Pixels along the [x-axis, y-axis] or samples
        self.sizePx = [sizePx[0]*self.mm, sizePx[1]*self.mm] # pixel size [x-axis, y-axis]

        self.sizeSensorX = self.numPx[0]*self.sizePx[0] # sensor size along x-axis
        self.sizeSensorY = self.numPx[1]*self.sizePx[1] # sensor size along y-axis

        # Sampling plane 2x for linearization
        self.Sx = 2*self.sizeSensorX
        self.Sy = 2*self.sizeSensorY

        self.samplingRate = 1/self.sizePx[0]

        # max freq to prevent aliasing by the transfer function
        self.delU = 1/self.Sx
        self.delV = 1/self.Sy

        self.freqRows = np.linspace(-1/(2*self.sizePx[0]), 1/(2*self.sizePx[0]), int(1/(self.sizePx[0]*self.delU))) ## -maxFreq/2, +maxFreq/2
        self.freqCols = np.linspace(-1/(2*self.sizePx[1]), 1/(2*self.sizePx[1]), int(1/(self.sizePx[1]*self.delV)))

        self.u, self.v = np.meshgrid(self.freqRows, self.freqCols)
        k = self.wl**(-2) - self.u**2 - self.v**2

        # removing evanscent waves: limiting transfer function does that but to remove 
        # numerical errors that may occur, safer to zero freqs > 1/wl
        if not np.all(k > 0):
            k[k < 0] = 0

        self.w = np.sqrt(k) # freq_z

        self.wavefield_z0 = None
        self.wavefield_z1 = None

        self.fft_wave_z0 = None
        self.fft_wave_z1 = None
        self.z = None # distance to propagate wavefield at z0 to parallel plane at z1

        # same optical axis as self.wavefield_z1 and self.wavefield_z0 at z1
        # self.uOblique = np.copy(self.u) # w/o x and y ticks spectrum at z1 oblique should look the same as at not-oblique
        # self.vOblique = np.copy(self.v)
    
    def propogate(self, dist: float):
        assert self.wavefield_z0 is not None, "Use method wavefied first"
        assert self.fft_wave_z0 is not None, "Use method wavefied first"

        self.z = dist*self.mm # distance to propogate along the z axis
        self.uLimit = 1/(np.sqrt((2*self.delU*self.z)**2 + 1)* self.wl)
        self.vLimit = 1/(np.sqrt((2*self.delV*self.z)**2 + 1)* self.wl)
        
        H = np.exp(1j*2*np.pi*self.w*self.z)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask = np.ones_like(H)
        mask[np.logical_or(np.abs(self.u) > int(self.uLimit), np.abs(self.v) >= int(self.vLimit))] = 0

        H = mask*H
        self.fft_wave_z1 = H*self.fft_wave_z0

        self.wavefield_z1 = np.fft.ifft2(np.fft.fftshift(self.fft_wave_z1))
        self.wavefield_z1 = self.wavefield_z1[
            int(self.fft_wave_z1.shape[0]/2 - self.wavefield_z0.shape[0]/2):int(self.fft_wave_z0.shape[0]/2 + self.wavefield_z0.shape[0]/2),
            int(self.fft_wave_z1.shape[1]/2 - self.wavefield_z0.shape[1]/2):int(self.fft_wave_z0.shape[1]/2 + self.wavefield_z0.shape[1]/2)    
            ]
        
        return self.wavefield_z1

    def wavefield(self, wave: np.array(np.complex128)):
        assert [wave.shape[1], wave.shape[0]] == self.numPx, "Incorrect number of pixels specified in constructor"
        # not using this wavefield to calculate here for speed as function may be used repeatedly

        linImg = np.zeros([int(self.Sy/self.sizePx[1]), int(self.Sx/self.sizePx[0])], dtype=np.complex128) # creates zeros of the size of the sensor
        linImg[int(linImg.shape[0]/2 - wave.shape[0]/2):int(linImg.shape[0]/2 + wave.shape[0]/2), 
            int(linImg.shape[1]/2 - wave.shape[1]/2):int(linImg.shape[1]/2 + wave.shape[1]/2)] = wave # ensures the wave is at the center of linImg
 
        self.wavefield_z0 = wave
        self.fft_wave_z0 = np.fft.fftshift(np.fft.fft2(linImg))

    def setup_limit_info(self):
        assert self.z != None, "Distance is set to none"
        ## Nice to know 1: max freq and angle that the camera/plane can record without aliasing
        maxFreqPossible = self.samplingRate/2
        maxAnglePossible = self.samplingRate*self.wl # cosine

        ## Nice to know 2: max freq and angle that the setup allows. Freqs above these will not
        ## reach the next plane
        maxAngleSetupX = self.sizeSensorX/(2*np.linalg.norm([self.z, self.sizeSensorX])) # cosine \thetaX
        maxAngleSetupY = self.sizeSensorY/(2*np.linalg.norm([self.z, self.sizeSensorY])) # cosing \thetaY

        maxFreqSetupX = maxAngleSetupX/self.wl
        maxFreqSetupY = maxAngleSetupY/self.wl

        print(f'Max Freq and Angle the camera/plane can record without aliasing: {maxFreqPossible*self.mm} cycles/mm | {maxAnglePossible} radians')
        
        print(f'Max Freq and Angle the setup allows (freqs > do not reach the next plane): {(maxFreqSetupX*self.mm, maxFreqSetupY*self.mm)} cycles/mm | {(maxAngleSetupX, maxAngleSetupY)} radians')

    
    def visualizations(self):
        """
        Too many repeated lines of code in the main scripts.
        Should be able to plot the spectrums with correct ticks.
        """

        pass
    

if __name__=='__main__':
    pass