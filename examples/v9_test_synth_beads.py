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

def generate_ref_obj_wavefields_from_vol_v2(RI_distribution, phase_mask, dist_m_im, dist_m_im_var, 
                                            wl, n_background, spatial_resolution, spatial_support, 
                                            mag=60, focal_plan_var=2e-6, padding=256, NA=0.85,
                                            partial_coherence_smoothing=0.1, im_to_ob_space_scale=30):
    
    k = 2*np.pi/wl
    # im_to_ob_space_scale = 30
    mag2 = int(mag/im_to_ob_space_scale)
    
    
    # prop though volume and then to focal plane
    output_field = propagator.propagate_beam_2(np.ones_like(RI_distribution[..., 1]), RI_distribution, n_background, wl, spatial_resolution)
    output_field = propagator.propagate(output_field, wl/n_background, spatial_resolution, spatial_support[2]/2, padding=padding, direction='backward')
    
    # for focal stacks
    if focal_plan_var > 0:
        output_field = propagator.propagate(output_field, wl, spatial_resolution, np.abs(focal_plan_var), padding=padding, direction='forward')
    else:
        output_field = propagator.propagate(output_field, wl, spatial_resolution, np.abs(focal_plan_var), padding=padding, direction='backward')

    # NA based low-pass filtering
    output_field = utils.low_pass_filter_NA(output_field, wl, spatial_resolution, NA)
    
    # approx partial coherence
    output_field = gaussian_filter(output_field, partial_coherence_smoothing, truncate=10)
    
    # magnification
    dx, dy = output_field.shape

    # output_field_b_sc_mag = cv2.resize(output_field_b_sc.real[int(dx/2 - dx/(2*mag2)):int(dx/2 + dx/(2*mag2)), int(dy/2 - dy/(2*mag2)):int(dy/2 + dy/(2*mag2))], 
                                    # (dx, dy), interpolation = cv2.INTER_CUBIC) + 1j*cv2.resize(output_field_b_sc.imag[int(dx/2 - dx/(2*mag2)):int(dx/2 + dx/(2*mag2)), int(dy/2 - dy/(2*mag2)):int(dy/2 + dy/(2*mag2))], (dx, dy), interpolation = cv2.INTER_CUBIC)
    output_field = cv2.resize(output_field.real, (dx*mag2, dy*mag2), interpolation = cv2.INTER_CUBIC) + 1j*cv2.resize(output_field.imag, (dx*mag2, dy*mag2), interpolation = cv2.INTER_CUBIC)
    
    # prop to phase mask plane
    output_field = propagator.propagate(output_field, wl, np.array(spatial_resolution)*im_to_ob_space_scale, dist_m_im + dist_m_im_var, padding=padding, direction='backward')
    # visualization.visualize_complex_field(output_field, np.array(spatial_support)*im_to_ob_space_scale)
    
    # calculate ground truth flow
    flow_y, flow_x = utils.grad_optr(np.angle(output_field))
    gt_flow_y = (np.remainder(flow_y + np.pi, 2*np.pi) - np.pi)/((spatial_resolution[0]*im_to_ob_space_scale)**2)/k*(dist_m_im + dist_m_im_var)
    gt_flow_x = (np.remainder(flow_x + np.pi, 2*np.pi) - np.pi)/(spatial_resolution[1]*im_to_ob_space_scale)**2/k*(dist_m_im + dist_m_im_var)

    # mask modulation and prop to image plane [1:-1, 1:-1]
    output_field_sensor = propagator.propagate(output_field*phase_mask, wl, np.array(spatial_resolution)*im_to_ob_space_scale, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    # output_field_sensor = propagator.propagate(output_field*phase_mask, wl, np.array(spatial_resolution)*30, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    
    # mask modulation and prop to image plane
    ref_field_sensor = propagator.propagate(phase_mask, wl, np.array(spatial_resolution)*im_to_ob_space_scale, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    # ref_field_sensor = propagator.propagate(phase_mask, wl, np.array(spatial_resolution)*30, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    
    return ref_field_sensor[5:-5, 5:-5], output_field_sensor[5:-5, 5:-5], [-1*gt_flow_x[5:-5, 5:-5], -1*gt_flow_y[5:-5, 5:-5]]


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


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # optimization
    losses = []
    epochs = args.samples

    # Grid and propagation parameters setup
    wl = 640e-9
    spatial_resolution = [100e-9, 100e-9, 100e-9] # dx, dy, dz
    grid_shape = [500, 500, 200] # x=0->, y=0->, z=0->
    n_background = 1.518 # immersion medium RI
    spatial_support = [spatial_resolution[i]*grid_shape[i] for i in range(3)]

    c_m = [25e-6, 25e-6, 10e-6]
    c_v = [5e-6, 5e-6, 2e-6]
    
    rad_params = [3, 8, 1e-6]
    RI_params = [1.5, 0.07]
    
    pad = 256
    NA = 0.85
    mag = 60
    im_to_ob_space_scale = 30
    mag2 = int(mag/im_to_ob_space_scale)
    dist_m_im = 1.43e-3
    partial_coherence_smoothing = 0.001

    side_length = 10e-6
    tile_size = int(side_length/(spatial_resolution[0]*im_to_ob_space_scale))
    RI_pm = 1.46
    pm_smoothing = 3

    int_gaus_noise = 2e-3

    geom = geometry.Geometry(grid_shape, spatial_resolution, n_background)
    transforms = utils.OpticalFlowTransformRAFT()
    
    wc_reconstructor = cws_module.CWS()

    losses_wc = []
    losses_NN = []
    # runs/dd_mm/model_data_v
    date = f'{datetime.today()}'.split()[0]

    for it1 in range(epochs):
        
        # generate phase mask
        phase_mask = create_phase_mask([2, 4], np.array(grid_shape)*mag2, tile_size, wl, RI_pm, 
                                smoothing=pm_smoothing, padding=1)[:-1, :-1]
        
        model.eval()
        with torch.no_grad():
            
            # generate data 3D tensor
            num_elements = np.random.randint(1, 6)
            print(f'{it1}:{num_elements}')
            
            RI_distribution_1 = geometry.generate_bead_data(geom, c_m, c_v, rad_params, RI_params, num_elements).get_grid()
            geom.reset_grid()
            
            # generate NN input and label
            dist_m_im_var = np.random.rand()*2e-3 - 0.5e-3
            f_plane_delta = np.random.randn()*4e-6
            
            ref_wave, obj_wave, gt_flow = generate_ref_obj_wavefields_from_vol_v2(RI_distribution_1, phase_mask, dist_m_im, dist_m_im_var, 
                                            wl, n_background, spatial_resolution, spatial_support, 
                                            mag=mag, focal_plan_var=f_plane_delta, padding=pad, NA=NA,
                                            partial_coherence_smoothing=partial_coherence_smoothing, 
                                            im_to_ob_space_scale=im_to_ob_space_scale)
            
            img_ref = torch.clamp(torch.abs(torch.tensor(ref_wave[::2, ::2]))**2 + int_gaus_noise*torch.randn(495, 495), min=0).float()
            img_obj = torch.clamp(torch.abs(torch.tensor(obj_wave[::2, ::2]))**2 + int_gaus_noise*torch.randn(495, 495), min=0).float()

            img1_batch, img2_batch = utils.preprocess([img_ref/torch.mean(img_ref)], [img_obj/torch.mean(img_obj)], transforms)
            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
        
            predicted_flows = F.resize(list_of_flows[-1].detach().cpu(), size=(img_obj.shape))
            opd_NN = 1*utils.int_2d_fourier([predicted_flows[0, 0], predicted_flows[0, 1]], (6e-6)**2/(dist_m_im+dist_m_im_var))/(2*np.pi)
            
            _, _, loss_wc = wc_reconstructor.run(10000*np.array(img_ref).astype(np.float64), 
                    10000*np.array(img_obj).astype(np.float64), 
                    prior=args.prior if args.prior is not None else [100, 1000, 100, 5], 
                    iter=args.iter if args.iter is not None else [15, 10, 10], tol=1e-4)
            _, opd_wc = wc_reconstructor.get_field(pixel_size=6e-6, z=dist_m_im+dist_m_im_var, RI=2.)
            opd_wc = -1*opd_wc
            
            gt_flow = [gt_flow[0][::2, ::2], gt_flow[1][::2, ::2]]
            gt = 1*(6e-6)**2/(dist_m_im+dist_m_im_var)*utils.int_2d_fourier(gt_flow, 1)/(2*np.pi)
            
            losses_wc.append(np.linalg.norm(gt*1e6 - opd_wc*1e6))
            losses_NN.append(np.linalg.norm(gt*1e6 - opd_NN*1e6))
            
            if args.data_save == 'yes':
                ckpt_name = f"{ckpt[5:-4].replace('/', '_')}_ds_synth_beads"
                if not os.path.exists(f'runs/test/{ckpt_name}/data'):
                    os.makedirs(f'runs/test/{ckpt_name}/data')
                    
                plt.imshow(np.hstack([gt*1e6, opd_NN*1e6, opd_wc*1e6]), cmap='jet', vmin=-0.25, vmax=0.25)
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
