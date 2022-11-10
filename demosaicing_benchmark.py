# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:40:46 2022

@author: jamyl
"""
import os
import random
import logging

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.utils_image import computePSNR
import colour_demosaicing
import demosaicnet

# demosaicnet is messing with logging somehow
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.WARNING)
# plt_logger = logging.getLogger('plt')
# plt_logger.setLevel(logging.WARNING)


DATASET_PATH = Path("P:/Kodak")

#%% synthetic burst functions

def decimate(burst):
    output = np.empty((burst.shape[0], burst.shape[1], burst.shape[2]))
    output[:,::2,::2] = burst[:,::2,::2,2] #b
    output[:,::2,1::2] = burst[:,::2,1::2,1] #g 01
    output[:,1::2,::2] = burst[:,1::2,::2,1] #g10
    output[:,1::2,1::2] = burst[:,1::2,1::2,0] #r]

    return output

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

#%%
CFA = np.array([[2, 1], [1, 0]]) #convention for bayer in synthetic burst
transformation_params = {'max_translation':3,
                          'max_shear': 0,
                          'max_ar_factor': 0,
                          'max_rotation': 0}

params = {'block matching': {
                'mode':'bayer',
                'tuning': {
                    # WARNING: these parameters are defined fine-to-coarse!
                    'factors': [1, 2, 2, 4],
                    'tileSizes': [16, 16, 16, 8],
                    'searchRadia': [1, 4, 4, 4],
                    'distances': ['L1', 'L2', 'L2', 'L2'],
                    # if you want to compute subpixel tile alignment at each pyramid level
                    'subpixels': [False, True, True, True]
                    }},
            'kanade' : {
                'mode':'bayer',
                'epsilon div' : 1e-6,
                'tuning' : {
                    'tileSize' : 16,
                    'tileSize Block Matching':16,
                    'kanadeIter': 6, # 3 
                    }},
            'robustness' : {
                'exif':{'CFA Pattern':CFA},
                'mode':'bayer',
                'on':False,
                'tuning' : {
                    'tileSize': 16,
                    't' : 0,            # 0.12
                    's1' : 2,          #12
                    's2' : 12,              # 2
                    'Mt' : 0.8,         # 0.8
                    }
                },
            'merging': {
                'exif':{'CFA Pattern':CFA},
                'mode':'bayer',
                'kernel':'handheld',
                'scale': 1,
                'tuning': {
                    'tileSize': 16,
                    'k_detail' : 0.33, # [0.25, ..., 0.33]
                    'k_denoise': 5,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    }
                }}
params['robustness']['std_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_std_ISO_50.npy')
params['robustness']['diff_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_diff_ISO_50.npy')
options = {'verbose' : 0}

N_images = len(os.listdir(DATASET_PATH))
PSNR = {"handheld": np.zeros(N_images),
        "malvar" : np.zeros(N_images),
        "bilinear" : np.zeros(N_images),
        "mosaicnet" : np.zeros(N_images)}

SSIM = {"handheld": np.zeros(N_images),
        "malvar" : np.zeros(N_images),
        "bilinear" : np.zeros(N_images),
        "mosaicnet" : np.zeros(N_images)}

demosaicnet_bayer = demosaicnet.BayerDemosaick()

#%% 
for im_id, filename in tqdm(enumerate(os.listdir(DATASET_PATH)), total=N_images):
    impath = DATASET_PATH/filename
    
    ground_truth = plt.imread(impath.as_posix()).astype(np.float64)
    #  ensuring the image can be decimated into Bayer
    if ground_truth.shape[0]%2 == 1:
        ground_truth = ground_truth[:-1,:]
    if ground_truth.shape[1]%2 == 1:
        ground_truth = ground_truth[:,:-1]
        
    burst, _ = single2lrburst(ground_truth, 15, downsample_factor=1, transformation_params=transformation_params)
    dec_burst = decimate(burst).astype(np.float32)
        
    with torch.no_grad():
        mosaic = demosaicnet.bayer(np.transpose(burst[0], [2, 0, 1]))
        mosaicnet_output = demosaicnet_bayer(torch.from_numpy(mosaic).unsqueeze(0)).squeeze(0).cpu().numpy()
    mosaicnet_output = np.clip(mosaicnet_output, 0, 1).transpose(1,2,0).astype(np.float64)
    # mosaicnet is outputing a cropped image. We need to crop ground truth
    crop = int((ground_truth.shape[0] - mosaicnet_output.shape[0])/2)
    
    handheld_output = main(dec_burst[0], dec_burst[1:], options, params)[0][:, :, :3].astype(np.float64)
    malvar_output = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(dec_burst[0], pattern='BGGR')
    bilinear_output = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(dec_burst[0], pattern='BGGR')

    
    PSNR["handheld"][im_id] = computePSNR(ground_truth[crop:-crop, crop:-crop], handheld_output[crop:-crop, crop:-crop])
    PSNR["malvar"][im_id] = computePSNR(ground_truth[crop:-crop, crop:-crop], malvar_output[crop:-crop, crop:-crop])
    PSNR["bilinear"][im_id] = computePSNR(ground_truth[crop:-crop, crop:-crop], bilinear_output[crop:-crop, crop:-crop])
    
    SSIM["handheld"][im_id] = ssim(ground_truth[crop:-crop, crop:-crop], handheld_output[crop:-crop, crop:-crop], channel_axis=2)
    SSIM["malvar"][im_id] = ssim(ground_truth[crop:-crop, crop:-crop], malvar_output[crop:-crop, crop:-crop], channel_axis=2)
    SSIM["bilinear"][im_id] = ssim(ground_truth[crop:-crop, crop:-crop], bilinear_output[crop:-crop, crop:-crop], channel_axis=2)
    
    PSNR["mosaicnet"][im_id] = computePSNR(ground_truth[crop:-crop, crop:-crop], mosaicnet_output)
    SSIM["mosaicnet"][im_id] = ssim(ground_truth[crop:-crop, crop:-crop], mosaicnet_output, channel_axis=2)
    
    
    print('\nPSNR')
    print('\tHandheld : {}'.format( PSNR["handheld"][im_id]))
    print('\tmalvar : {}'.format( PSNR["malvar"][im_id]))
    print('\tbilinear : {}'.format( PSNR["bilinear"][im_id]))
    print('\tmosaicnet : {}'.format( PSNR["mosaicnet"][im_id]))
    
    print('\nSSIM')
    print('\tHandheld : {}'.format( SSIM["handheld"][im_id]))
    print('\tmalvar : {}'.format( SSIM["malvar"][im_id]))
    print('\tbilinear : {}'.format( SSIM["bilinear"][im_id]))
    print('\tmosaicnet : {}'.format( SSIM["mosaicnet"][im_id]))
    

#%% ploting result

PSNR_means = [np.mean(PSNR[method]) for method in PSNR.keys()]
SSIM_means = [np.mean(SSIM[method]) for method in PSNR.keys()]

plt.figure("PSNR")
plt.bar(PSNR.keys(), PSNR_means)
plt.title("PSNR")

plt.figure("SSIM")
plt.bar(SSIM.keys(), SSIM_means)
plt.title("SSIM")
    