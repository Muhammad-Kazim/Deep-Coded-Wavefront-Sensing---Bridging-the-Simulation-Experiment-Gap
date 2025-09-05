import os
import sys
sys.path.insert(0, "/home/u491036/Projects/2025_Optimizing_Phase_Mask/coded_wfs_sim")

from coded_wfs_sim import geometry
from coded_wfs_sim import propagator
# from coded_wfs_sim import visualization
from coded_wfs_sim import utils

import numpy as np
from tifffile import tifffile
from matplotlib import pyplot as plt
import cv2

import torch
from torch import nn, Tensor
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
from torch.utils.tensorboard import SummaryWriter

from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import correlate2d, convolve2d, tukey
# from scipy.interpolate import RegularGridInterpolator
import scipy

from zernike import RZern

from typing import Optional, Union
from datetime import datetime
import argparse
import cws_module


def generate_cell(HEK_params, cart, meshgrid, mean_var=5e-4, std_var=5, ran_crop=True, ran_flip=True, ran_rotate=True):
    
    temp_coeffs = std_var*HEK_params[1]*np.random.rand(861) + HEK_params[0] + mean_var*np.random.rand(861)
    gen_cell = np.where(cart.eval_grid(temp_coeffs, matrix=True) < 0.005, 0., cart.eval_grid(temp_coeffs, matrix=True))
    gen_cell = np.where(meshgrid[0]**2 + meshgrid[1]**2 > 0.5, 0., gen_cell)
    
    if ran_crop:
        # random crop -> displace the cell and change the size of the cell
        ch = 50 + np.random.randint(-50, 100)
        cv = -50 + np.random.randint(-100, 50)
        
        gen_cell = gen_cell[ch:cv, ch:cv]
        
        crop_odd = (ch -1*cv) % 2 == 1
        pad_by = np.ceil((ch -1*cv)/2).astype(int)
        if crop_odd:
            gen_cell = gen_cell[:-1, :-1]
            
        gen_cell = np.pad(gen_cell, pad_by)
        
    if ran_flip:
        gen_cell = np.flip(gen_cell, axis=np.random.randint(0, 2))
        gen_cell = np.flip(gen_cell, axis=np.random.randint(0, 2))
        
    if ran_rotate:
        gen_cell = scipy.ndimage.rotate(gen_cell, np.random.randint(-45, 45), reshape=False)
    
    return gen_cell


def generate_ref_obj_wavefields_HEK_cells(synth_cell, phase_mask, dist_m_im, dist_m_im_var,
                                wl, spatial_resolution, f_plane_delta=30e-6, padding=256, conv_opt_dx=3):
    """Propagate the synthetic cell through the phase mask and record the object and ref wavefields

    Args:
        synth_cell (array): OPD of generated cell [meters]
        phase_mask (array): phase mask  in radians
        dist_m_im (float): nominal disance bw mask and sensor
        dist_m_im_var (float): variation in mask-sensor distance
        wl (float): wavelength
        spatial_resolution (list):resolution oof simulation in image space
        f_plane_delta (float, optional): Distance planar cell travel to have amplitude features. Defaults to 3e-3.
        padding (int, optional): _description_. Defaults to 256.

    Returns:
        ref_image_sensor (float): magnitude of ref speckle field
        obj_image_sensor (gloat): magnitude of object speckle field
        gt_flow (list): scaled gradients of object
    """
    
    k = 2*np.pi/wl

    # adds features to the amp which show up in the intensity but also add diffraction due to prop
    output_field = propagator.propagate(np.exp(1j*k*synth_cell), wl, np.array(spatial_resolution)/10, f_plane_delta, 
                                        padding=padding, direction='forward')
    # pass through radially reducing RI in a multislice model for circular object to simulate less bend on the edges and more in the center.

    # from image plane prop back to mask plane from image plane (plus variation) ensures focus plane imaging
    output_field = propagator.propagate(output_field, wl, spatial_resolution, dist_m_im + dist_m_im_var, 
                                            padding=padding, direction='backward')
    output_field = gaussian_filter(output_field, np.random.randint(2, 4, size=1)[0]) # smoothing -> partial spatial coherence. Also, gradient spreads, beocmes visible in amplitude
    
    # ground truth flow
    flow_y, flow_x = utils.grad_optr(np.angle(output_field))
    gt_flow_y = (np.remainder(flow_y + np.pi, 2*np.pi) - np.pi)
    gt_flow_x = (np.remainder(flow_x + np.pi, 2*np.pi) - np.pi)
    
    # aerial to sensor flow
    conv_opt = np.ones([conv_opt_dx, conv_opt_dx])
    gt_flow_y = scipy.signal.convolve2d(gt_flow_y, conv_opt, mode='same')[::conv_opt_dx, ::conv_opt_dx]
    gt_flow_x = scipy.signal.convolve2d(gt_flow_x, conv_opt, mode='same')[::conv_opt_dx, ::conv_opt_dx]
    
    gt_flow_y = gt_flow_y/(conv_opt_dx*spatial_resolution[0])**2/k*(dist_m_im + dist_m_im_var)
    gt_flow_x = gt_flow_x/(conv_opt_dx*spatial_resolution[1])**2/k*(dist_m_im + dist_m_im_var)
    
    # mask modulation and prop to image plane
    output_field = propagator.propagate(output_field*phase_mask, wl, spatial_resolution, dist_m_im + dist_m_im_var, 
                                        padding=padding, direction='forward')
    ref_field = propagator.propagate(phase_mask, wl, spatial_resolution, dist_m_im + dist_m_im_var, 
                                        padding=padding, direction='forward')
    
    # aerial to sensor image
    ref_image_sensor = scipy.signal.convolve2d(np.abs(ref_field)**2, 
                                               conv_opt, mode='same')[::conv_opt_dx, ::conv_opt_dx]
    obj_image_sensor = scipy.signal.convolve2d(np.abs(output_field)**2, 
                                               conv_opt, mode='same')[::conv_opt_dx, ::conv_opt_dx] 
    
    # normalize and add gaussian with sigma 1.5/median 1 before inpit to network to match collected images
    ref_image_sensor = median_filter(ref_image_sensor/ref_image_sensor.max(), 1)
    obj_image_sensor = median_filter(obj_image_sensor/obj_image_sensor.max(), 1)
              
    return ref_image_sensor[5:-5, 5:-5], obj_image_sensor[5:-5, 5:-5], [gt_flow_x[5:-5, 5:-5], gt_flow_y[5:-5, 5:-5]]


def create_phase_mask(height_range, grid_shape, tile_size, wl, RI_pm, smoothing=5, padding=0):
    h_pm = wl/np.random.randint(height_range[0], height_range[1])
    hmap_pm = geometry.initialize_hmap_uniform_sampling(grid_shape[:2], tile_size, h_pm)
    hmap_pm = np.pad(hmap_pm, padding, 'edge')
    opd_pm = gaussian_filter(hmap_pm*RI_pm + (hmap_pm.max() - hmap_pm)*1., smoothing) # phase mask in air
    
    return np.exp(1j*(2*np.pi/wl)*opd_pm)


# model
def init_model_RAFT(model, device='cuda', checkpoint=None):
    
    if checkpoint == None:
        print('Loading models with pretrained weights')
        weights = Raft_Large_Weights.DEFAULT.get_state_dict()
        weights['feature_encoder.convnormrelu.0.weight'] = torch.mean(weights['feature_encoder.convnormrelu.0.weight'], dim=1).unsqueeze(1)
        weights['context_encoder.convnormrelu.0.weight'] = torch.mean(weights['context_encoder.convnormrelu.0.weight'], dim=1).unsqueeze(1)
    else:
        print(f'Loading models with checkpoint: {checkpoint}')
        weights = torch.load(checkpoint, map_location=torch.device(device))
    
    # model = raft_large(progress=False)
    model.feature_encoder.convnormrelu[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.context_encoder.convnormrelu[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    
    model.load_state_dict(weights)
    
    return model.to(device)
    

def RAFT_loss(predictions, target, device='cuda'):
	loss_fn = torch.nn.L1Loss()
	N = len(predictions)
	
	w = torch.pow(torch.tensor(0.8), N-torch.tensor(range(1, N+1))).to(device)
	loss_n = torch.sum(torch.abs(torch.stack(predictions, dim=0) - target.unsqueeze(0).repeat(12, 1, 1, 1, 1)), dim=[1, 2, 3, 4])
	
	return torch.sum(w*loss_n).squeeze()


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_load", type=str, help='relative skpt path', required=True)
    parser.add_argument("--data_save",  choices=['yes', 'no'], default='no')

    parser.add_argument('--samples', type=int, help='total epochs', required=True)
    parser.add_argument('--seed', type=int, help='seed', default=1)
    
    parser.add_argument('--iter', nargs='+', type=int, help='total amp phasae')
    parser.add_argument('--prior', nargs='+', type=float, help='total amp phasae')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # initializations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    ckpt_path = None
    if args.ckpt_load != 'no':
        ckpt = args.ckpt_load
        ckpt_path = f"/home/u491036/Projects/2025_Optimizing_Phase_Mask/coded_wfs_sim/examples/{ckpt}"
    model = init_model_RAFT(raft_large(progress=False), device=device, checkpoint=ckpt_path)


    # optimization
    losses = []
    epochs = args.samples
    
    print('Loading Zernike polynomials')
    cart = RZern(40)
    L, K = 800, 800
    ddx = np.linspace(-1.0, 1.0, L)
    ddy = np.linspace(-1.0, 1.0, K)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    
    HEK_mean = scipy.io.loadmat(f'./HEK_synth_data_generstor.mat')['mean'].ravel()
    HEK_std = scipy.io.loadmat(f'./HEK_synth_data_generstor.mat')['std'].ravel()

    # Grid and propagation parameters setup
    wl = 640e-9
    spatial_resolution = [2e-6, 2e-6, 2e-6] # dx, dy, dz
    spatial_support = [spatial_resolution[i]*[L, K][i] for i in range(2)]
    
    dist_m_im = 2e-3 # meters
    pad = 256
    
    side_length = 10e-6
    tile_size = int(side_length/(spatial_resolution[0]))
    RI_pm = 1.46
    pm_smoothing = 4
    
    int_gaus_noise = 2e-3
    
    transforms = utils.OpticalFlowTransformRAFT()
    wc_reconstructor = cws_module.CWS()
    
    losses_wc = []
    losses_NN = []
    # runs/dd_mm/model_data_v
    date = f'{datetime.today()}'.split()[0]

    for it1 in range(epochs):
        
        # generate phase mask
        phase_mask = create_phase_mask([3, 6], (L, K), tile_size, wl, RI_pm, smoothing=pm_smoothing)
        synth_cell_opd = 1e-6*generate_cell([HEK_mean, HEK_std], cart, [xv, yv], mean_var=5e-4, std_var=8, ran_crop=False, ran_flip=False, ran_rotate=False) # shape [L, K]
        
        model.eval()
        with torch.no_grad():
            
            # generate NN input and label
            dist_m_im_var = np.random.randn()*1e-3
            f_plane_delta = np.random.randint(20, 40, size=1)[0]*1e-6
            ref_img, obj_img, gt_flow = generate_ref_obj_wavefields_HEK_cells(synth_cell_opd, phase_mask, dist_m_im, dist_m_im_var, wl, spatial_resolution, f_plane_delta=f_plane_delta, padding=pad)
            
            img_ref = torch.clamp(torch.abs(torch.tensor(ref_img)) + int_gaus_noise*torch.randn(ref_img.shape), min=0).float()
            img_obj = torch.clamp(torch.abs(torch.tensor(obj_img)) + int_gaus_noise*torch.randn(ref_img.shape), min=0).float()
            
            img1_batch, img2_batch = utils.preprocess([img_ref/torch.mean(img_ref)], [img_obj/torch.mean(img_obj)], transforms)
            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
        
            predicted_flows = F.resize(list_of_flows[-1].detach().cpu(), size=(img_obj.shape))
            opd_NN = -1*utils.int_2d_fourier([predicted_flows[0, 0], predicted_flows[0, 1]], (6e-6)**2/(dist_m_im+dist_m_im_var))/(2*np.pi)
            
            _, _, loss_wc = wc_reconstructor.run(10000*np.array(img_ref).astype(np.float64), 
                    10000*np.array(img_obj).astype(np.float64), 
                    prior=args.prior if args.prior is not None else [100, 1000, 100, 5], 
                    iter=args.iter if args.iter is not None else [15, 10, 10], tol=1e-4)
            _, opd_wc = wc_reconstructor.get_field(pixel_size=6e-6, z=dist_m_im+dist_m_im_var, RI=2.)
            # opd_wc = -1*opd_wc
            
            gt = 1*(6e-6)**2/(dist_m_im+dist_m_im_var)*utils.int_2d_fourier(gt_flow, 1)/(2*np.pi)
            
            losses_wc.append(np.linalg.norm(gt*1e6 - opd_wc*1e6))
            losses_NN.append(np.linalg.norm(gt*1e6 - opd_NN*1e6))
            
            if args.data_save == 'yes':
                ckpt_name = f"{ckpt[5:-4].replace('/', '_')}_ds_synth_cell"
                if not os.path.exists(f'runs/test/{ckpt_name}/data'):
                    os.makedirs(f'runs/test/{ckpt_name}/data')
                    
                plt.imshow(np.hstack([gt*1e6, opd_NN*1e6, opd_wc*1e6]), cmap='jet', vmin=-0.1, vmax=0.25)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f'runs/test/{ckpt_name}/data/seed{args.seed}_{it1}.png', pad_inches=0, bbox_inches='tight')
                plt.close()
                
            # Explicit cleanup
            del img1_batch, img2_batch, list_of_flows
            torch.cuda.empty_cache()

        
    plt.plot(losses_wc, 'r*', label='CCWFS')
    plt.plot(losses_NN, 'b*', label='DCWFS')
    plt.title(f'RAFT:{np.array(losses_NN).mean()/opd_NN.size:4f}um, WC:{np.array(losses_wc).mean()/opd_wc.size:4f}um')
    plt.savefig(f'runs/test/{ckpt_name}/data/loss.png', pad_inches=0, bbox_inches='tight')

plt.legend()
