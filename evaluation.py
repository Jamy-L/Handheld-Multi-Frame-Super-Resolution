# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:14:52 2022

@author: jamyl
"""


from time import time
from tqdm import tqdm
import random

import numpy as np
from numba import cuda, float64
import torch
import cv2
import matplotlib.pyplot as plt
from skimage.transform import warp 

from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.block_matching import alignBurst
from handheld_super_resolution.optical_flow import get_closest_flow, lucas_kanade_optical_flow



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
            shift = (downsample_factor / 2.0) - 0.5
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

            max_rotation = transformation_params.get('max_rotation', 0.0)
            #theta = 5
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
    if burst.shape[1]%2 == 0:
        croped_burst = burst
    else:
        croped_burst = burst[:, :-1, :,:]
    if burst.shape[2]%2 == 1:
        croped_burst = croped_burst[:,:,:-1,:]
        

    output = np.empty((croped_burst.shape[0], croped_burst.shape[1], croped_burst.shape[2]), dtype = np.uint16)
    output[:,::2,::2] = croped_burst[:,::2,::2,2] #b
    output[:,::2,1::2] = croped_burst[:,::2,1::2,1] #g 01
    output[:,1::2,::2] = croped_burst[:,1::2,::2,1] #g10
    output[:,1::2,1::2] = croped_burst[:,1::2,1::2,0] #r
    return output


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

    options = {'verbose' : 2}
    pre_alignment, aligned_tiles = alignBurst(dec_burst[0], dec_burst[1:],params['block matching'], options)
    pre_alignment = pre_alignment[:, :, :, ::-1]
    tile_size = aligned_tiles.shape[-1]
    print("TS ", tile_size)


    lk_alignment = lucas_kanade_optical_flow(
        dec_burst[0], dec_burst[1:], pre_alignment, options, params['kanade'], debug=True)
    lk_alignment[-1][ :, :, :, -2:]/=2 #last alignment is multiplied by 2 by the LK flow function
    for i,x in enumerate(lk_alignment):
        lk_alignment[i][ :, :, :, -2:] = 2*x[:, :, :, -2:]
    
    augmented_pre_alignment = np.zeros(lk_alignment[0].shape) #from pure translation to complex homography
    augmented_pre_alignment[:,:,:,-2:] = pre_alignment
    lk_alignment.insert(0, augmented_pre_alignment)


    imsize = (dec_burst.shape[1], dec_burst.shape[2])
    
    upscaled = np.empty((len(lk_alignment), dec_burst.shape[0]-1, dec_burst.shape[1], dec_burst.shape[2], 2))
    for i in range(len(lk_alignment)):
        upscaled[i] = upscale_alignement(np.array(lk_alignment[i]), imsize, tile_size)
    # we need to upscale because estimated_al is patchwise

    return lk_alignment, upscaled

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

def warp_flow(image, flow, rgb=False):
    Y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    X = np.linspace(0, image.shape[1] - 1, image.shape[1])
    Xm, Ym = np.meshgrid(X, Y)
    Z = np.stack((Xm, Ym)) #[2, imshape], X first, Y second
    Z = (Z + flow.transpose(2,0,1))[::-1,:,:] #[2, imshape], Y first X second
    if rgb : 
        r = warp(image[:,:,0], inverse_map=Z)
        g = warp(image[:,:,1], inverse_map=Z)
        b = warp(image[:,:,2], inverse_map=Z)
        warped = np.stack((r,g,b)).transpose(1,2,0)
    else:
        warped = warp(image[:,:], inverse_map=Z)
    return warped

def im_SE(ground_truth, warped):
    return np.mean(ground_truth - warped, axis=2)**2

def im_MSE(ground_truth, warped):
    return np.mean(im_SE(ground_truth, warped))


def evaluate_alignment(comp_alignment, comp_imgs, ref_img, label="", imshow=False):
    """
    

    Parameters
    ----------
    alignment : Array [n_iter, n_images, imsize_y, imsize_x, 2]
        DESCRIPTION.
    imshow : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    warped_images = np.empty(comp_alignment.shape[:-1]+(3,))
    
    im_EQ = np.empty(comp_alignment.shape[:-1])
    print("Evaluating {}".format(label))
    for image_index in tqdm(range(comp_alignment.shape[1])):
        for iteration in range(comp_alignment.shape[0]):
            warped_images[iteration, image_index] = warp_flow(comp_imgs[image_index],
                                                              comp_alignment[iteration, image_index], rgb=True)
            im_EQ[iteration, image_index] = im_SE(ref_img,
                                                  warped_images[iteration, image_index])

    last_im_MSE = np.mean(im_EQ[-1])
    if comp_alignment.shape[0] > 1:
        # plt.figure("flow MSE")
        # plt.plot([i for i in range(len(flow_EQ))], np.mean(flow_EQ, axis=(1,2,3)), label=label)
        # plt.xlabel('lk iteration')
        # plt.ylabel('MSE on flow')
        # plt.legend()
        
        plt.figure("image MSE")
        plt.plot([i for i in range(len(im_EQ))], np.mean(im_EQ, axis=(1,2,3)), label=label)
        plt.xlabel('lk iteration')
        plt.ylabel('MSE on warped image')
        plt.legend()
    
        plt.figure("flow norm")
        plt.plot([np.mean(np.linalg.norm(comp_alignment[i], axis=3)) for i in range(comp_alignment.shape[0])], label=label)
        plt.xlabel('lk iteration')
        plt.ylabel('mean norm of optical flow')
        plt.legend()
    
        plt.figure("flow step")
        plt.plot([np.mean(np.linalg.norm(2*comp_alignment[i+1] - 2*comp_alignment[i], axis=3)) for i in range(comp_alignment.shape[0]-1)], label=label)
        plt.xlabel('lk iteration')
        plt.ylabel('mean norm of optical flow step for each iteration')
        plt.legend()
    else : #Farneback
        # plt.figure("flow MSE")
        # plt.plot([8], [last_flow_MSE], marker = 'x', label = "Farneback")
        # plt.legend()
        
        plt.figure("image MSE")
        plt.plot([8], [last_im_MSE], marker = 'x', label = "Farneback")
        plt.legend()
        
        
    if imshow : 
        for i in range(im_EQ.shape[0]):
            # plt.figure("{} alignment, step {}".format(label, i))
            # plt.imshow(np.log10(np.mean(flow_EQ[i], axis = 0)), cmap = "Reds")
            # plt.colorbar()
            
            plt.figure("{} warped alignment, step {}".format(label, i))
            plt.imshow(np.log10(im_EQ[i,1]),vmin = -3, vmax=4, cmap = "Reds")
            plt.colorbar()

    return warped_images, im_EQ 

#%%
# #Warning : tileSize is expressed in terms of grey pixels.
# CFA = np.array([[2, 1], [1, 0]])

# params = {'block matching': {
#                 'mode':'bayer2',
#                 'tuning': {
#                     # WARNING: these parameters are defined fine-to-coarse!
#                     'factors': [1, 2, 2, 2],
#                     'tileSizes': [16, 16, 16, 8],
#                     'searchRadia': [1, 4, 4, 4],
#                     'distances': ['L1', 'L2', 'L2', 'L2'],
#                     # if you want to compute subpixel tile alignment at each pyramid level
#                     'subpixels': [False, True, True, True]
#                     }},
#             'kanade' : {
#                 'mode':'bayer2',
#                 'epsilon div' : 1e-6,
#                 'tuning' : {
#                     'tileSizes' : 16,
#                     'kanadeIter': 6, # 3 
#                     }},
#             'robustness' : {
#                 'exif':{'CFA Pattern':CFA},
#                 'mode':'bayer2',
#                 'tuning' : {
#                     'tileSizes': 16,
#                     't' : 0,            # 0.12
#                     's1' : 2,          #12
#                     's2' : 12,              # 2
#                     'Mt' : 0.8,         # 0.8
#                     'sigma_t' : 0.03,
#                     'dt' : 1e-3,
#                     }
#                 },
#             'merging': {
#                 'exif':{'CFA Pattern':CFA},
#                 'mode':'bayer2',
#                 'scale': 2,
#                 'tuning': {
#                     'tileSizes': 16,
#                     'k_detail' : 0.33, # [0.25, ..., 0.33]
#                     'k_denoise': 5,    # [3.0, ...,5.0]
#                     'D_th': 0.05,      # [0.001, ..., 0.010]
#                     'D_tr': 0.014,     # [0.006, ..., 0.020]
#                     'k_stretch' : 4,   # 4
#                     'k_shrink' : 2,    # 2
#                     }
#                 }}
# params['robustness']['std_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_std_ISO_50.npy')
# params['robustness']['diff_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_diff_ISO_50.npy')
# options = {'verbose' : 3}

# img = plt.imread("P:/DIV2K_valid_HR/DIV2K_valid_HR/0900.png")*255
# #img = plt.imread("P:/Urban100_SR/image_SRF_4/img_040_SRF_4_HR.png")*255
# transformation_params = {'max_translation':10,
#                           'max_shear': 0,
#                           'max_ar_factor': 0,
#                           'max_rotation': 3}
# burst, flow = single2lrburst(img, 10, downsample_factor=2, transformation_params=transformation_params)
# # flow is unussable because it is pointing from moving frame to ref. We would need the opposite


# dec_burst = (decimate(burst)/255).astype(np.float32)

# grey_burst = np.mean(burst, axis = 3)/255

# #%%
# params["block matching"]["mode"] = 'grey'
# params["kanade"]["mode"] = 'grey'
# pre_alignment, _ = alignBurst(grey_burst[0], grey_burst[1:2],params['block matching'], options)
# pre_alignment = pre_alignment[:,:,:,::-1]
# lk_alignment = lucas_kanade_optical_flow(grey_burst[0], grey_burst[1:2],
#                                          pre_alignment, options, params['kanade']).copy_to_host()

# pre_al = np.zeros(pre_alignment.shape[:-1] + (6,))
# pre_al[:,:,:,-2:] = pre_alignment

# imsize = grey_burst[0].shape[:2]
# upscaled_al = upscale_alignement(lk_alignment, imsize, params['block matching']['tuning']['tileSizes'][0])
# warped = warp_flow(grey_burst[1], upscaled_al[0], rgb=False)
# plt.figure("grey warped")
# plt.imshow(warped, cmap='gray')

# #%% testing pipleine on grey images
# params["block matching"]["mode"] = 'grey'
# params["kanade"]["mode"] = 'grey'
# params["merging"]["mode"] = 'grey'
# params["robustness"]["mode"] = 'grey'
# output, R, r, alignment = main(grey_burst[0]/255, grey_burst[1:]/255, options, params)
# plt.figure("merge on grey images")
# plt.imshow(output[:,:,0], cmap='gray')
# plt.figure("ref")
# plt.imshow(grey_burst[0]/255, cmap="gray")

# #%% same with bayer
# params["block matching"]["mode"] = 'bayer'
# params["kanade"]["mode"] = 'bayer'
# pre_alignment, _ = alignBurst(dec_burst[0], dec_burst[1:2], params['block matching'], options)
# pre_alignment = pre_alignment[:,:,:,::-1]
# lk_alignment = lucas_kanade_optical_flow(dec_burst[0], dec_burst[1:2],
#                                          pre_alignment, options, params['kanade']).copy_to_host()

# pre_al = np.zeros(pre_alignment.shape[:-1] + (6,))
# pre_al[:,:,:,-2:] = pre_alignment
# imsize = dec_burst[0].shape[:2]

# upscaled_al = upscale_alignement(pre_al, imsize, 2*params['block matching']['tuning']['tileSizes'][0])
# warped = warp_flow(burst[1]/255, upscaled_al[0], rgb=True)
# plt.figure('bayer warp')
# plt.imshow(warped)

# #%% testing pipleine on one bayer image
# params["block matching"]["mode"] = 'bayer'
# params["kanade"]["mode"] = 'bayer'
# params["merging"]["mode"] = 'bayer'
# params["robustness"]["mode"] = 'bayer'

# output, R, r, alignment = main(dec_burst[0], dec_burst[1:], options, params)
# plt.figure("merge on bayer images")
# plt.imshow(output[:,:,:3])
# plt.figure("ref")
# plt.imshow(burst[0]/255)

# #%% aligning LK on bayer
# params["block matching"]["mode"] = 'bayer'
# params["kanade"]["mode"] = 'bayer'
# raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params)
# t1 = time()
# fb_alignment = align_fb(dec_burst)
# print('farneback evaluated : ', time()-t1)

# #%% evaluating lk bayer
# lk_warped_images, lk_im_EQ = evaluate_alignment(upscaled_lk_alignment, burst[1:], burst[0],  label = "LK", imshow=False)
# fb_warped_images, fb_im_EQ = evaluate_alignment(fb_alignment[None], burst[1:], burst[0], label = "FarneBack", imshow=True)


# #%% ploting burst
# plt.figure("ref")
# plt.imshow(burst[0]/255)
# for i in range(4):
#     plt.figure("{}".format(i))
#     plt.imshow(burst[i+1]/255)

# #%% matching warps with original

# plt.figure("ref")
# plt.imshow(burst[0]/255)
# for i in range(lk_warped_images.shape[1]):
#     plt.figure("{}".format(i))
#     plt.imshow(lk_warped_images[-1, i]/255)


# #%%
# plt.figure("LK")
# plt.imshow(lk_warped_images[-1, 0]/255)
# plt.figure("Farneback")
# plt.imshow(fb_warped_images[0, 0]/255)
# plt.figure("Block Matching")
# plt.imshow(lk_warped_images[0,0]/255)



