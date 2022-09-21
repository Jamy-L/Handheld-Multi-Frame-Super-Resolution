# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:14:52 2022

@author: jamyl
"""

import numpy as np
import torch
import random
import cv2
import matplotlib.pyplot as plt
from block_matching import alignHdrplus
from optical_flow import get_closest_flow, lucas_kanade_optical_flow
from numba import cuda, float64
from time import time



#%% Single img to burst
# generates a downsampled synthetic burst from a single image 
# with the optical flow from the ref img to the moved images.

def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    """ Generates a transformation matrix corresponding to the input transformation parameters """
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])

    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0],
                        [0.0, 0.0, 1.0]])

    t_mat = t_scale @ t_rot @ t_shear @ t_mat

    t_mat = t_mat[:2, :]

    return t_mat


def single2lrburst(image, burst_size, downsample_factor=1, transformation_params=None,
                   interpolation_type='bilinear'):
    """ Generates a burst of size burst_size from the input image by applying random transformations defined by
    transformation_params, and downsampling the resulting burst by downsample_factor.
    args:
        image - input sRGB image
        burst_size - Number of images in the output burst
        downsample_factor - Amount of downsampling of the input sRGB image to generate the LR image
        transformation_params - Parameters of the affine transformation used to generate a burst from single image
        interpolation_type - interpolation operator used when performing affine transformations and downsampling
    """

    if interpolation_type == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_type == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError

    normalize = False
    if isinstance(image, torch.Tensor):
        if image.max() < 2.0:
            image = image * 255.0
            normalize = True
        image = np.array((image)).astype(np.uint8)

    burst = []
    sample_pos_inv_all = []

    rvs, cvs = torch.meshgrid([torch.arange(0, image.shape[0]),
                               torch.arange(0, image.shape[1])])

    sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()

    for i in range(burst_size):
        if i == 0:
            # For base image, do not apply any random transformations. We only translate the image to center the
            # sampling grid
            shift = (downsample_factor / 2.0) - 0.5# I dont understant this line
            translation = (shift, shift)
            theta = 0.0
            shear_factor = (0.0, 0.0)
            scale_factor = (1.0, 1.0)
        else:
            # Sample random image transformation parameters
            max_translation = transformation_params.get('max_translation', 0.0)

            if max_translation <= 0.01:
                shift = (downsample_factor / 2.0) - 0.5
                translation = (shift, shift)
            else:
                translation = (random.uniform(-max_translation, max_translation),
                                random.uniform(-max_translation, max_translation))
                # translation = (30, 30)

            max_rotation = transformation_params.get('max_rotation', 0.0)
            theta = random.uniform(-max_rotation, max_rotation)

            max_shear = transformation_params.get('max_shear', 0.0)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            shear_factor = (shear_x, shear_y)

            max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
            ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

            max_scale = transformation_params.get('max_scale', 0.0)
            scale_factor = np.exp(random.uniform(-max_scale, max_scale))

            scale_factor = (scale_factor, scale_factor * ar_factor)

        output_sz = (image.shape[1], image.shape[0])

        # Generate a affine transformation matrix corresponding to the sampled parameters
        t_mat = get_tmat((image.shape[0], image.shape[1]), translation, theta, shear_factor, scale_factor)
        t_mat_tensor = torch.from_numpy(t_mat)

        # Apply the sampled affine transformation
        image_t = cv2.warpAffine(image, t_mat, output_sz, flags=interpolation,
                                 borderMode=cv2.BORDER_CONSTANT)

        t_mat_tensor_3x3 = torch.cat((t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0)
        t_mat_tensor_inverse = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

        sample_pos_inv = torch.mm(sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()).view(
            *sample_grid.shape[:2], -1)

        if transformation_params.get('border_crop') is not None:
            border_crop = transformation_params.get('border_crop')

            image_t = image_t[border_crop:-border_crop, border_crop:-border_crop, :]
            sample_pos_inv = sample_pos_inv[border_crop:-border_crop, border_crop:-border_crop, :]

        # Downsample the image
        image_t = cv2.resize(image_t, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                             interpolation=interpolation)
        sample_pos_inv = cv2.resize(sample_pos_inv.numpy(), None, fx=1.0 / downsample_factor,
                                    fy=1.0 / downsample_factor,
                                    interpolation=interpolation)

        sample_pos_inv = torch.from_numpy(sample_pos_inv).permute(2, 0, 1)

        if normalize:
            image_t = torch.from_numpy(image_t).float() / 255.0
        else:
            image_t = torch.from_numpy(image_t).float()
        burst.append(image_t)
        sample_pos_inv_all.append(sample_pos_inv / downsample_factor)

    burst_images = torch.stack(burst)
    sample_pos_inv_all = torch.stack(sample_pos_inv_all)

    # Compute the flow vectors to go from the i'th burst image to the base image
    flow_vectors = -(sample_pos_inv_all - sample_pos_inv_all[:1, ...])

    return np.array(burst_images), np.array(flow_vectors)





def decimate(burst):
    output = np.empty((burst.shape[0], burst.shape[1], burst.shape[2]), dtype = np.uint16)
    output[:,::2,::2] = burst[:,::2,::2,2] #b
    output[:,::2,1::2] = burst[:,::2,1::2,1] #r 01
    output[:,1::2,::2] = burst[:,1::2,::2,1] #r10
    output[:,1::2,1::2] = burst[:,1::2,1::2,0] #b
    return output

def get_im_mse(burst, alignment):
    
    
    n_images, imsize_y, imsize_x, _ = burst.shape
    tile_size = 32
    imsize = (imsize_y, imsize_x)
    n_images -= 1
    iters = alignment.shape[0]
    
    @cuda.jit
    def pixel_mse(burst, alignment,  im_mse):
        s = cuda.shared.array(1, dtype=float64)
        s[0] = 0
        acc = cuda.shared.array(1, dtype=float64)
        acc[0] = 0
        i, idy, idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        
        im_id = cuda.threadIdx.x
        
        local_flow = cuda.local.array(2, dtype=float64)
        get_closest_flow(idx, idy, alignment[i, im_id], tile_size, imsize, local_flow)
        
        new_idx = idx + local_flow[0]
        new_idy = idy + local_flow[1]
        
        if 0<= new_idx < imsize_x and 0 <= new_idy < imsize_y:
            cuda.atomic.add(s, 0, 
                            (burst[0, idy, idx, 0] - burst[im_id + 1, round(new_idy), round(new_idx), 0])**2)
            cuda.atomic.add(s, 0, 
                            (burst[0, idy, idx, 1] - burst[im_id + 1, round(new_idy), round(new_idx), 1])**2)
            cuda.atomic.add(s, 0, 
                            (burst[0, idy, idx, 2] - burst[im_id + 1, round(new_idy), round(new_idx), 2])**2)
            cuda.atomic.add(acc, 0, 3)
        
        cuda.syncthreads()
        if im_id == 0: #single threaded
            if acc[0] == 0:
                im_mse[i, idy, idx] = 0
            else:
                im_mse[i, idy, idx] = s[0]/acc[0]
            
    
    
    im_mse = cuda.device_array((iters, imsize_y, imsize_x), dtype = np.float32)
    blockspergrid = (iters, imsize_y, imsize_x)
    pixel_mse[blockspergrid, (n_images)](burst, alignment, im_mse)
    return im_mse.copy_to_host()


def upscale_alignement(alignment, imsize, tile_size):
    upscaled_alignment = cuda.device_array(((alignment.shape[0],)+imsize+(2,)))
    cuda_alignment = cuda.to_device(np.ascontiguousarray(alignment))
    @cuda.jit
    def get_flow(upscaled_alignment, pre_alignment):
        im_id, x, y=cuda.grid(3)
        if 0 <= x <imsize[1] and 0 <= y <imsize[0] and 0 <=  im_id < pre_alignment.shape[0]:
            local_flow = cuda.local.array(2, dtype=float64)
            get_closest_flow(x, y, pre_alignment[im_id], tile_size, imsize, local_flow)
            upscaled_alignment[im_id, y, x, 0] = local_flow[0]
            upscaled_alignment[im_id, y, x, 1] = local_flow[1]

        
    threadsperblock = (2, 16, 16)
    blockspergrid_n = int(np.ceil(alignment.shape[0]/threadsperblock[0]))
    blockspergrid_x = int(np.ceil(imsize[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(imsize[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_n, blockspergrid_x, blockspergrid_y)
    
    get_flow[blockspergrid, threadsperblock](upscaled_alignment, cuda_alignment)
    return upscaled_alignment.copy_to_host()

def align_lk(dec_burst, params):

    options = {'verbose' : 3}
    pre_alignment, aligned_tiles = alignHdrplus(dec_burst[0], dec_burst[1:],params['block matching'], options)
    pre_alignment = pre_alignment[:, :, :, ::-1]
    tile_size = aligned_tiles.shape[-1]



    lk_alignment = lucas_kanade_optical_flow(
        dec_burst[0], dec_burst[1:], pre_alignment, options, params['kanade'], debug=True)
    lk_alignment[-1]/=2 #last alignment is multiplied by 2 by the LK flow function
    
    for i,x in enumerate(lk_alignment):
        lk_alignment[i] = 2*x
        
    lk_alignment.insert(0, pre_alignment)

    imsize = (dec_burst.shape[1], dec_burst.shape[2])
    
    upscaled = np.empty((len(lk_alignment), dec_burst.shape[0]-1, dec_burst.shape[1], dec_burst.shape[2], 2))
    for i in range(len(lk_alignment)):
        upscaled[i] = upscale_alignement(np.array(lk_alignment[i]), imsize, tile_size)
    # we need to upscale because estimated_al is patchwise

    return upscaled

def align_fb(dec_burst):
    ref_grey = np.empty((int(dec_burst.shape[1]/2), int(dec_burst.shape[2]/2)))
    ref_grey[:,:] = (dec_burst[0, ::2, ::2] + dec_burst[0, 1::2, ::2] + dec_burst[0, ::2, 1::2] + dec_burst[0,1::2, 1::2])/4  


    farnback_flow = np.empty((dec_burst.shape[0] - 1, int(dec_burst.shape[1]/2), int(dec_burst.shape[2]/2), 2))

    for i in range(dec_burst.shape[0] - 1):
         
        # Capture another frame and convert to gray scale
        
        comp_grey = np.empty((int(dec_burst.shape[1]/2), int(dec_burst.shape[2]/2)))
        comp_grey[:,:] = (dec_burst[i+1, ::2, ::2] + dec_burst[i+1, 1::2, ::2] + dec_burst[i+1, ::2, 1::2] + dec_burst[i+1,1::2, 1::2])/4  
        
        
        # Optical flow is now calculated
        farnback_flow[i] = cv2.calcOpticalFlowFarneback(ref_grey, comp_grey, None, 0.5, 3, 16, 3, 5, 1.2, 0)
    upscaled_fb = np.empty( (dec_burst.shape[0] - 1, ) + dec_burst.shape[1:] + (2, ))
    upscaled_fb[:, ::2, ::2, :] = farnback_flow
    upscaled_fb[:, 1::2, ::2, :] = farnback_flow
    upscaled_fb[:, ::2, 1::2, :] = farnback_flow
    upscaled_fb[:, 1::2, 1::2, :] = farnback_flow
    return upscaled_fb*2 # greys are twice smaller

def evaluate_alignment(alignment, ground_truth, label="", imshow=False):
    """
    

    Parameters
    ----------
    alignment : Array [n_iter, n_images, imsize_y, imsize_x, 2]
        DESCRIPTION.
    ground_truth : TYPE
        DESCRIPTION.
    imshow : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    def flow_quad_error(ground_truth, alignment):
        return np.linalg.norm(ground_truth - alignment, axis=4)


    EQ = flow_quad_error(ground_truth, alignment)
    if alignment.shape[0] > 1:
        plt.figure("flow MSE")
        plt.plot([i for i in range(len(EQ))], np.mean(EQ, axis=(1,2,3)))
        plt.xlabel('lk iteration')
        plt.ylabel('MSE on flow')
    
        plt.figure("flow norm")
        plt.plot([np.mean(np.linalg.norm(alignment[i], axis=3)) for i in range(alignment.shape[0])])
        plt.xlabel('lk iteration')
        plt.ylabel('mean norm of optical flow')
    
        plt.figure("flow step")
        plt.plot([np.mean(np.linalg.norm(2*alignment[i+1] - 2*alignment[i], axis=3)) for i in range(alignment.shape[0]-1)])
        plt.xlabel('lk iteration')
        plt.ylabel('mean norm of optical flow step for each iteration')
        
    print("Last mean norm of optical flow on {} : {}".format(label, np.mean(np.linalg.norm(alignment[-1], axis=3))))
        
        
    if imshow : 
        for i, e in enumerate(EQ):
            plt.figure("{} alignment, step {}".format(label, i))
            plt.imshow(np.mean(e, axis = 0), vmin=0, vmax =3, cmap = "Reds")
            plt.colorbar()

    print("Last flow MSE on {} : {}".format(label, np.mean(EQ[-1])))
    
    #im_mse = get_im_mse(burst, np.array(lk_alignment))

    # for i, mse in enumerate(im_mse):
    #     plt.figure("lk alignment {} Image MSE ".format(i))
    #     plt.imshow(mse, vmin=0, vmax =100, cmap = "Reds")
    #     plt.colorbar()    

    # plt.figure("Image MSE")
    # plt.plot(np.mean(im_mse, axis = (1, 2)))


#%%
params = {'block matching': {
                'mode':'bayer',
                'tuning': {
                    # WARNING: these parameters are defined fine-to-coarse!
                    'factors': [1, 2, 2, 2],
                    'tileSizes': [16, 16, 16, 8],
                    'searchRadia': [1, 4, 4, 4],
                    'distances': ['L1', 'L2', 'L2', 'L2'],
                    # if you want to compute subpixel tile alignment at each pyramid level
                    'subpixels': [False, True, True, True]
                    }},
            'kanade' : {
                'epsilon div' : 1e-6,
                'tuning' : {
                    'tileSizes' : 32,
                    'kanadeIter': 7, # 3 
                    }},
            'merging': {
                'scale': 2,
                'tuning': {
                    'tileSizes': 32,
                    'k_detail' : 0.3,  # [0.25, ..., 0.33]
                    'k_denoise': 4,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    't' : 0.12,        # 0.12
                    's1' : 2,
                    's2' : 2,
                    'Mt' : 0.8,
                    'sigma_t' : 2,
                    'dt' : 15},
                    }
            }

img = plt.imread("P:/DIV2K_valid_HR/DIV2K_valid_HR/0806.png")*255
transformation_params = {'max_translation':10}
burst, flow = single2lrburst(img, 5, downsample_factor=2, transformation_params=transformation_params)
flow = flow[1:].transpose(0, 2, 3, 1)


plt.figure("ref")
plt.imshow(burst[0]/255)
plt.figure("1")
plt.imshow(burst[1]/255)

dec_burst = decimate(burst)

lk_alignment = align_lk(dec_burst, params)
t1 = time()
fb_alignment = align_fb(dec_burst)
print('farneback evaluated : ', time()-t1)


evaluate_alignment(lk_alignment, flow, label = "LK", imshow=True)

evaluate_alignment(fb_alignment[None], flow, label = "FarneBack", imshow=True)

origin = flow[0]
lk = lk_alignment[-1, 0]
fb = fb_alignment[0]

