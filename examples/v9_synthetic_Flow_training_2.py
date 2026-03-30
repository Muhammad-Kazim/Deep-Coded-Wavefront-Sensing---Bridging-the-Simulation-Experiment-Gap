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
# from scipy.signal import correlate2d, convolve2d, tukey
# from scipy.interpolate import RegularGridInterpolator

from typing import Optional, Union
from datetime import datetime
import argparse


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
    # gt_flow_y = (np.remainder(flow_y + np.pi, 2*np.pi) - np.pi)/((spatial_resolution[0]*im_to_ob_space_scale)**2)/k*(dist_m_im + dist_m_im_var)
    # gt_flow_x = (np.remainder(flow_x + np.pi, 2*np.pi) - np.pi)/(spatial_resolution[1]*im_to_ob_space_scale)**2/k*(dist_m_im + dist_m_im_var)
    gt_flow_y = median_filter(flow_y, 3)/((spatial_resolution[0]*im_to_ob_space_scale)**2)/k*(dist_m_im + dist_m_im_var)
    gt_flow_x = median_filter(flow_x, 3)/(spatial_resolution[1]*im_to_ob_space_scale)**2/k*(dist_m_im + dist_m_im_var)

    # mask modulation and prop to image plane [1:-1, 1:-1]
    output_field_sensor = propagator.propagate(output_field*phase_mask, wl, np.array(spatial_resolution)*im_to_ob_space_scale, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    # output_field_sensor = propagator.propagate(output_field*phase_mask, wl, np.array(spatial_resolution)*30, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    
    # mask modulation and prop to image plane
    ref_field_sensor = propagator.propagate(phase_mask, wl, np.array(spatial_resolution)*im_to_ob_space_scale, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    # ref_field_sensor = propagator.propagate(phase_mask, wl, np.array(spatial_resolution)*30, dist_m_im + dist_m_im_var, padding=padding, direction='forward')
    
    return ref_field_sensor[5:-5, 5:-5], output_field_sensor[5:-5, 5:-5], [-1*gt_flow_x[5:-5, 5:-5], -1*gt_flow_y[5:-5, 5:-5]]


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
    
    parser.add_argument("--psd", type=str, help='presampled data path or no', default='no')
    parser.add_argument("--ckpt_load", type=str, help='relative skpt path or no', default='no')
    parser.add_argument("--ckpt_save",  choices=['yes', 'no'], default='no')
    parser.add_argument("--data_save",  choices=['yes', 'no'], default='no')

    parser.add_argument("--loss",  choices=['l1', 'l2'], default='l2')
    
    parser.add_argument('--epochs', type=int, help='total epochs', required=True)
    parser.add_argument('--iter_pm', type=int, help='number of iters with same phase mask', required=True)
    
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)

    args = parser.parse_args()
    
    np.random.seed(0)
    
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

    # runs/dd_mm/model_data_v
    date = f'{datetime.today()}'.split()[0]
    writer = SummaryWriter(f'runs/{date}/{args.name}')

    # validation 4 images
    ref_val = []
    obj_val = []
    for i in ['001', '002', '004']:
        im_ref = torch.tensor(tifffile.imread(f'data/validation/ref_{i}.tif')).float()
        im_obj = torch.tensor(tifffile.imread(f'data/validation/obj_{i}.tif')).float()
        
        if im_ref.dim() > 2:
            im_ref = im_ref[..., 0]
            im_obj = im_obj[..., 0]
        
        ref_val.append(F.center_crop(im_ref, (480, 640))/torch.mean(im_ref))
        obj_val.append(F.center_crop(im_obj, (480, 640))/torch.mean(im_obj))
    
    img1_batch_val, img2_batch_val = utils.preprocess(ref_val, obj_val, transforms)
    # img1_batch_val, img2_batch_val = img1_batch_val.to(device), img2_batch_val.to(device)
    
    if args.psd != 'no':
        print('Loading data list')
        data_list = os.listdir(f'{args.psd}/obj')
    for it1 in range(epochs):
        
        if args.psd != 'no':
            np.random.shuffle(data_list)
        # generate phase mask
        phase_mask = create_phase_mask([2, 4], np.array(grid_shape)*mag2, tile_size, wl, RI_pm, 
                                smoothing=pm_smoothing, padding=1)[:-1, :-1]
        
        ref_imgs = []
        obj_imgs = []
        gt_flows = []
        
        model.train()
        # s_crop_dims = np.random.randint(400, high=475, size=2)
        for it2 in range(update_wegiths_iter):
            
            if args.psd == 'no':
                # generate data 3D tensor
                num_elements = np.random.randint(1, 6)
                print(f'{it1}:{it2}-{num_elements}')
            
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
                gt_flows.append([gt_flow[0][::2, ::2], gt_flow[1][::2, ::2]])
            
            else:
                img_ref = tifffile.imread(f'{args.psd}/ref/{data_list[it2]}')/10e3
                img_obj = tifffile.imread(f'{args.psd}/obj/{data_list[it2]}')/10e3
                gt_flow0 = tifffile.imread(f'{args.psd}/grad/{data_list[it2][:-5]}_0.tiff')
                gt_flow1 = tifffile.imread(f'{args.psd}/grad/{data_list[it2][:-5]}_1.tiff')
                
                img_ref = torch.clamp(torch.tensor(img_ref) + int_gaus_noise*torch.randn(495, 495), min=0).float()
                img_obj = torch.clamp(torch.tensor(img_obj) + int_gaus_noise*torch.randn(495, 495), min=0).float()
                
                gt_flows.append([gt_flow0/65000*64 - 32, gt_flow1/65000*64 - 32])
            
            # img_ref = F.center_crop(img_ref, (s_crop_dims[0], s_crop_dims[1]))
            # img_obj = F.center_crop(img_obj, (s_crop_dims[0], s_crop_dims[1]))
            
            ref_imgs.append(img_ref/torch.mean(img_ref))
            obj_imgs.append(img_obj/torch.mean(img_obj))    
            #gt_flows.append([gt_flow[0][::2, ::2], gt_flow[1][::2, ::2]])
            
            if args.data_save == 'yes':
                if not os.path.exists(f'runs/{date}/data/obj'):
                    os.makedirs(f'runs/{date}/data/obj')
                if not os.path.exists(f'runs/{date}/data/ref'):
                    os.makedirs(f'runs/{date}/data/ref')
                        
                if not os.path.exists(f'runs/{date}/data/ground_truth'):
                    os.makedirs(f'runs/{date}/data/ground_truth')
                    
                tifffile.imwrite(f'runs/{date}/data/obj/{it1}_{it2}.tiff', utils.normalization(img_obj))
                tifffile.imwrite(f'runs/{date}/data/ref/{it1}_{it2}.tiff', utils.normalization(img_ref))
                tifffile.imwrite(f'runs/{date}/data/ground_truth/{it1}_{it2}_0.tiff', utils.normalization(gt_flow[0][::2, ::2]))
                tifffile.imwrite(f'runs/{date}/data/ground_truth/{it1}_{it2}_1.tiff', utils.normalization(gt_flow[1][::2, ::2]))
                
                
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
        
        if it1 %  50 == 0:
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
