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

from scipy.ndimage import gaussian_filter
# from scipy.signal import correlate2d, convolve2d, tukey
# from scipy.interpolate import RegularGridInterpolator

from typing import Optional, Union
from datetime import datetime
import argparse


def generate_ref_obj_wavefields(RI_distribution, phase_mask, dist_m_im, dist_m_im_var,
                                wl, n_background, spatial_resolution, spatial_support, mag, f_plane_delta=0., padding=256, NA=1):

    k = 2*np.pi/wl
    output_field = propagator.propagate_beam_2(np.ones_like(RI_distribution[..., 1]), RI_distribution, n_background, wl, spatial_resolution)
    
    # prop to focal plane
    # f_plane_delta = moves the focus plane from RI_dist center
    output_field = propagator.propagate(output_field, wl/n_background, spatial_resolution, spatial_support[2]/2, padding=padding, direction='backward')
    
    if f_plane_delta < 0:
        output_field = propagator.propagate(output_field, wl/n_background, spatial_resolution, np.abs(f_plane_delta), padding=padding, direction='backward')
    else:
        output_field = propagator.propagate(output_field, wl/n_background, spatial_resolution, np.abs(f_plane_delta), padding=padding, direction='forward')
    
    # NA based low-pass filtering
    output_field = utils.low_pass_filter_NA(output_field, wl, spatial_resolution, 2*NA)
    
    # prop back to mask plane from image plane (plus variation) ensures focus plane imaging
    output_field = propagator.propagate(output_field, wl, np.array(spatial_resolution)*mag, dist_m_im + dist_m_im_var, 
                                        padding=padding, direction='backward')
    output_field = gaussian_filter(output_field, 2*np.random.rand(1)[0] + 1.) # smoothing -> partial spatial coherence
    # smoothing_param = np.random.randint(5, high=10, size=1)[0]
    
    # ground truth flow
    flow_y, flow_x = utils.grad_optr(np.angle(output_field))
    gt_flow_y = (np.remainder(flow_y + np.pi, 2*np.pi) - np.pi)/(spatial_resolution[0]*mag)**2/k*(dist_m_im + dist_m_im_var)
    gt_flow_x = (np.remainder(flow_x + np.pi, 2*np.pi) - np.pi)/(spatial_resolution[1]*mag)**2/k*(dist_m_im + dist_m_im_var)
    
    # mask modulation and prop to image plane
    output_field = propagator.propagate(output_field*phase_mask, wl, np.array(spatial_resolution)*mag, dist_m_im + dist_m_im_var, 
                                        padding=padding, direction='forward')
    ref_field = propagator.propagate(phase_mask, wl, np.array(spatial_resolution)*mag, dist_m_im + dist_m_im_var, 
                                        padding=padding, direction='forward')
    
    # return gaussian_filter(ref_field[5:-5, 5:-5], smoothing_param), gaussian_filter(output_field[5:-5, 5:-5], smoothing_param), [-1*gaussian_filter(gt_flow_x[5:-5, 5:-5], smoothing_param), -1*gaussian_filter(gt_flow_y[5:-5, 5:-5], smoothing_param)]
    
    return ref_field[5:-5, 5:-5], output_field[5:-5, 5:-5], [gt_flow_x[5:-5, 5:-5], gt_flow_y[5:-5, 5:-5]]


def create_phase_mask(height_range, grid_shape, tile_size, wl, RI_pm):
    h_pm = wl/np.random.randint(height_range[0], height_range[1])
    hmap_pm = geometry.initialize_hmap_uniform_sampling(grid_shape[:2], tile_size, h_pm)
    opd_pm = hmap_pm*RI_pm + (hmap_pm.max() - hmap_pm)*1. # phase mask in air
    
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
        weights = torch.load(checkpoint)
    
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

    parser.add_argument("--name", type=str, help='project/run/date/name', required=True)
    
    parser.add_argument("--ckpt_load", type=str, help='relative skpt path or no', default='no')
    parser.add_argument("--ckpt_save",  choices=['yes', 'no'], default='no')
    
    parser.add_argument("--loss",  choices=['l1', 'l2'], default='l2')
    
    parser.add_argument('--epochs', type=int, help='total epochs', required=True)
    parser.add_argument('--iter_pm', type=int, help='number of iters with same phase mask', required=True)
    
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)

    args = parser.parse_args()
    
    # initializations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    ckpt = None
    if args.ckpt_load != 'no':
        ckpt = args.ckpt_load
    model = init_model_RAFT(raft_large(progress=False), device=device, checkpoint=ckpt)

    learning_rate = args.lr

    if args.loss == 'l2':
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.L1Loss()
    # optimizer = torch.optim.Adam([u_est], lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # optimization
    losses = []
    epochs = args.epochs
    update_wegiths_iter = args.iter_pm

    # Grid and propagation parameters setup
    wl = 640e-9
    spatial_resolution = [200e-9, 200e-9, 200e-9] # dx, dy, dz
    grid_shape = [500, 500, 200] # x=0->, y=0->, z=0->
    n_background = 1.33 # immersion medium RI
    spatial_support = [spatial_resolution[i]*grid_shape[i] for i in range(3)]

    c_m = [50e-6, 50e-6, 20e-6]
    c_v = [15e-6, 15e-6, 3e-6]
    # c_v = [3e-6, 3e-6, 1e-6]
    rad_params = [8, 20, 1e-6]
    RI_params = [1.33, 0.1]

    dist_m_im = 2e-3 # meters
    pad = 512
    mag = 10

    side_length = 10e-6
    tile_size = int(side_length/(spatial_resolution[0]*mag))
    RI_pm = 1.46
    # h_pm = wl/(2*(RI_pm - 1.))

    int_gaus_noise = 1e-3

    geom = geometry.Geometry(grid_shape, spatial_resolution, n_background)
    transforms = utils.OpticalFlowTransformRAFT()

    # runs/dd_mm/model_data_v
    date = f'{datetime.today()}'.split()[0]
    writer = SummaryWriter(f'runs/{date}/{args.name}_{args.loss}')


    # validation 4 images
    ref_val = []
    obj_val = []
    for i in ['001', '002', '004']:
        im_ref = torch.tensor(tifffile.imread(f'data/validation/ref_{i}.tif')).float()
        im_obj = torch.tensor(tifffile.imread(f'data/validation/obj_{i}.tif')).float()
        
        if im_ref.dim() > 2:
            im_ref = im_ref[..., 0]
            im_obj = im_obj[..., 0]
        
        ref_val.append(F.center_crop(im_ref, (480, 640))/im_ref.max())
        obj_val.append(F.center_crop(im_obj, (480, 640))/im_obj.max())
        
    img1_batch_val, img2_batch_val = utils.preprocess(ref_val, obj_val, transforms)
    # img1_batch_val, img2_batch_val = img1_batch_val.to(device), img2_batch_val.to(device)

    def median_filter_2d_2ch(input_tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        Apply median filtering to a Bx2xHxW tensor (2 channels per image).

        Args:
            input_tensor: torch.Tensor of shape (B, 2, H, W)
            kernel_size: int, odd filter size

        Returns:
            torch.Tensor of shape (B, 2, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        B, C, H, W = input_tensor.shape
        assert C == 2, "Input must have 2 channels."

        pad = kernel_size // 2

        # Apply unfold to each channel separately
        filtered_channels = []
        for c in range(C):
            channel = input_tensor[:, c:c+1, :, :]  # shape: (B, 1, H, W)
            patches = torch.nn.functional.unfold(channel, kernel_size=kernel_size, padding=pad)  # shape: (B, K*K, H*W)
            median = patches.median(dim=1)[0]  # shape: (B, H*W)
            median = median.view(B, 1, H, W)   # shape: (B, 1, H, W)
            filtered_channels.append(median)

        # Concatenate filtered channels back along dim=1
        return torch.cat(filtered_channels, dim=1)  # shape: (B, 2, H, W)
    
    for it1 in range(epochs):
        
        # generate phase mask
        phase_mask = gaussian_filter(create_phase_mask([1, 3], grid_shape, tile_size, wl, RI_pm), 2*np.random.rand(1)[0] + 1.)
        
        ref_imgs = []
        obj_imgs = []
        gt_flows = []
        
        model.train()
        # s_crop_dims = np.random.randint(400, high=475, size=2)
        for it2 in range(update_wegiths_iter):
            
            # generate data 3D tensor
            num_elements = np.random.randint(1, 6)
            print(f'{it1}:{it2}-{num_elements}')
            
            
            RI_distribution_1 = geometry.generate_bead_data(geom, c_m, c_v, rad_params, RI_params, num_elements).get_grid()
            geom.reset_grid()
            # RI_distribution_2 = geometry.generate_bead_data(geom, c_m, c_v, rad_params, RI_params, num_elements).get_grid()
            # geom.reset_grid()
            
            # generate NN input and label
            dist_m_im_var = np.random.randn()*1e-3
            f_plane_delta = np.random.randn()*10e-6
            ref_wave, obj_wave, gt_flow = generate_ref_obj_wavefields(RI_distribution_1, phase_mask, dist_m_im, dist_m_im_var,
                                        wl, n_background, spatial_resolution, spatial_support, mag, f_plane_delta=f_plane_delta, padding=256, NA=np.random.randint(-4, 4)/10+1.)
            
            img_ref = torch.clamp(torch.abs(torch.tensor(ref_wave))**2 + int_gaus_noise*torch.randn(490, 490), min=0).float()
            img_obj = torch.clamp(torch.abs(torch.tensor(obj_wave))**2 + int_gaus_noise*torch.randn(490, 490), min=0).float()
            
            # img_ref = F.center_crop(img_ref, (s_crop_dims[0], s_crop_dims[1]))
            # img_obj = F.center_crop(img_obj, (s_crop_dims[0], s_crop_dims[1]))
            
            ref_imgs.append(img_ref/img_ref.max())
            obj_imgs.append(img_obj/img_obj.max())    
            gt_flows.append(gt_flow)
            # gt_flows.append([gt_flow[0][245 - int(s_crop_dims[0]/2):int(s_crop_dims[0]/2) + 245, 245 - int(s_crop_dims[1]/2):int(s_crop_dims[1]/2) + 245], gt_flow[1][245 - int(s_crop_dims[0]/2):int(s_crop_dims[0]/2) + 245, 245 - int(s_crop_dims[1]/2):int(s_crop_dims[1]/2) + 245]])   
            
        img1_batch, img2_batch = utils.preprocess(ref_imgs, obj_imgs, transforms)
        # img1_batch, img2_batch = F.gaussian_blur(img1_batch, 3, num_elements/10), F.gaussian_blur(img2_batch, 3, num_elements/10)
        # img1_batch = F.center_crop(img1_batch, (s_crop_dims[0], s_crop_dims[1]))
        # img2_batch = F.center_crop(img2_batch, (s_crop_dims[0], s_crop_dims[1]))
        list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            
        # predicted_flows = list_of_flows[-1].detach().cpu()
        targets = utils.process_labels(gt_flows)
        # targets = median_filter_2d_2ch(targets, kernel_size=15) # reduces coherence, also reduces max grads
        # targets = F.center_crop(targets, (s_crop_dims[0], s_crop_dims[1]))
        
        # loss = loss_fn(predicted_flows, targets.to(device))
        loss = RAFT_loss(list_of_flows, targets.to(device), device)
        print(f'{it1}:{loss.item()}')
        writer.add_scalar('Loss', loss.item(), it1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it1 %  10 == 0:
            flow_imgs = flow_to_image(list_of_flows[-1].detach().cpu())
            flow_targets = flow_to_image(targets.detach().cpu())
            grid = np.hstack([np.concatenate([img1, flow_img], axis=2) for (img1, flow_img) in zip(flow_targets, flow_imgs)])
            grid = F.resize(torch.tensor(grid).detach(), (600, 900))
            
            writer.add_image('train_images', grid, it1)
            
            model.eval()

            with torch.no_grad():
                list_of_flows_val = model(img1_batch_val.to(device), img2_batch_val.to(device))
                predicted_flows_val = list_of_flows_val[-1].detach().cpu()
                flow_imgs_val = flow_to_image(predicted_flows_val)
                #grid_val = np.hstack([np.concatenate([np.array(255*(img1.detach().cpu().repeat(3, 1, 1) + 1)/2, dtype=np.uint8), flow_img], axis=2) for (img1, flow_img) in zip(img2_batch_val, flow_imgs_val)])
                
                grid_val = np.hstack([np.concatenate([np.array(255*(img1.detach().cpu().repeat(3, 1, 1) + 1)/2, dtype=np.uint8), flow_img, np.repeat(np.expand_dims(utils.normalization(utils.int_2d_fourier([flows[0], flows[1]], 1), totype='int8'), axis=0), 3, axis=0)], axis=2) for (img1, flow_img, flows) in zip(img2_batch_val, flow_imgs_val, predicted_flows_val)])
                
                grid_val = F.resize(torch.tensor(grid_val).detach(), (600, 900))
                
                writer.add_image('valid_images', grid_val, it1)
            
            if args.ckpt_save == 'yes':
                if not os.path.exists(f'runs/{date}/ckpt'):
                    os.makedirs(f'runs/{date}/ckpt')
                torch.save(model.state_dict(), f'runs/{date}/ckpt/{args.name}_{args.loss}_{it1}.pth')
            
        writer.flush()
        
        # Explicit cleanup
        del img1_batch, img2_batch, targets, list_of_flows
        torch.cuda.empty_cache()

        ref_imgs.clear()
        obj_imgs.clear()
        gt_flows.clear()
                
        # losses.append(loss.item())
        
    writer.close()
