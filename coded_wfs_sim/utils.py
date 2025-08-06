import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as convolve
from scipy.fftpack import dct, idct

import torch
import torchvision.transforms.functional as F
from torch import nn, Tensor

def load_pkl(filename):
    """Reads in a geometry pickled object.

    Args:
        filename (str): path/to/file.pkl

    Returns:
        geometry: geometry object
    """
    
    if os.path.isfile(filename):
        print('Loading geometry object...')
        with open(filename, 'rb') as inp:
            geom = pickle.load(inp)
        
        return geom
    else:
        print('File does not exist.')
        

def normalization(field, totype='int16'):
    """normalizes field to 0-1.

    Args:
        field (float): 2d floating point arrays.
        totype (str): 'int16' or 'int8'
    """
    
    field = (field - field.min())/(field.max() - field.min())
    
    if totype == 'int16':
        return np.array(field*(2**16) - 1, dtype=np.uint16)
    elif totype == 'int8':
        return np.array(field*(2**8) - 1, dtype=np.uint8)
    else:
        print('Wrong totype')
        

def low_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
    fmax = NA/wl/2
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftfreq(wavefield.shape[0], dx)
    ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K > fmax] = 0.
    
    wave_spectrum = np.fft.fft2(wavefield)*mask
    
    return np.fft.ifft2(wave_spectrum)

# util functions

class OpticalFlowTransformRAFT(nn.Module):
    def forward(self, img1, img2):
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5], std=[0.5])
        img2 = F.normalize(img2, mean=[0.5], std=[0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[-1.0, 1.0]``."
        )
        
def preprocess(ref_img, obj_img, transforms):
    img1_batch = torch.stack(ref_img)
    img2_batch = torch.stack(obj_img)

    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)

    # print(img1_batch.shape, img2_batch.shape)

    return transforms(img1_batch.unsqueeze(1), img2_batch.unsqueeze(1))

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    
def process_labels(flows):
    
    imgs = []
    
    for im in range(len(flows)):
        imgs.append(F.resize(torch.tensor(np.array(flows[im])).float(), size=[520, 960], antialias=False))

    return torch.stack(imgs)

def grad_optr(image):

    dx = image[:, 1:] - image[:, :-1]
    dy = image[1:, :] - image[:-1, :] 
    
    dx = np.pad(dx, [[0, 0], [0, 1]], mode='edge')
    dy = np.pad(dy, [[0, 1], [0, 0]], mode='edge')

    return [dy, dx]


def freq_array(shape, sampling):
    f_freq_1d_x = np.fft.fftfreq(shape[0], sampling)
    f_freq_1d_y = np.fft.fftfreq(shape[1], sampling)
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y, indexing='ij')
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq

def int_2d_fourier(arr, sampling):
    freqs = freq_array(arr[0].shape, sampling)

    k_sq = np.where(freqs != 0, freqs**2, 1e-9)
    k = np.meshgrid(np.fft.fftfreq(arr[0].shape[0], sampling), np.fft.fftfreq(arr[0].shape[1], sampling), indexing='ij')

    v_int_x = np.real(np.fft.ifft2((np.fft.fft2(arr[1]) * k[0]) / (2*np.pi * 1j * k_sq)))
    v_int_y = np.real(np.fft.ifft2((np.fft.fft2(arr[0]) * k[1]) / (2*np.pi * 1j * k_sq)))

    v_int_fs = v_int_x + v_int_y
    return v_int_fs

def poisson_solver(gx, gy):
    """
    A DCT-based Poisson solver to integrate the surface from gradients.

    Parameters:
    - gx (np.ndarray): Gradient along the x-axis.
    - gy (np.ndarray): Gradient along the y-axis.

    Returns:
    - np.ndarray: Reconstructed surface.
    """
    # Pad size
    wid = 1
    gx = np.pad(gx, ((wid, wid), (wid, wid)))
    gy = np.pad(gy, ((wid, wid), (wid, wid)))
    
    # Define operators in the spatial domain
    nabla_x_kern = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    nabla_y_kern = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    
    # Define adjoint operator
    def nablaT(gx, gy):
        return convolve(gx, np.rot90(nabla_x_kern, 2), boundary='symmetric', mode='same') + \
               convolve(gy, np.rot90(nabla_y_kern, 2), boundary='symmetric', mode='same')

    # Generate inverse kernel
    H, W = gx.shape
    x_coord, y_coord = np.meshgrid(np.arange(W), np.arange(H))
    mat_x_hat = 2 * np.cos(np.pi * x_coord / W) + 2 * np.cos(np.pi * y_coord / H) - 4
    mat_x_hat[0, 0] = 1

    # Perform inverse filtering
    dct2 = lambda x: dct(dct(x.T, norm='ortho').T, norm='ortho')
    idct2 = lambda x: idct(idct(x.T, norm='ortho').T, norm='ortho')
    
    rec = idct2(dct2(nablaT(gx, gy)) / -mat_x_hat)
    rec = rec[wid:-wid, wid:-wid]
    rec = np.pad(rec[1:-1, 1:-1], ((1, 1), (1, 1)), mode='edge')
    rec -= np.mean(rec)
    
    return rec


if __name__=='__main__':
    pass