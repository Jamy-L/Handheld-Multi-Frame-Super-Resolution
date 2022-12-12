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
import colour_demosaicing
import math
from skimage import filters
import pandas as pd

from handheld_super_resolution.utils_image import compute_grey_images
from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.block_matching import alignBurst
from handheld_super_resolution.optical_flow import get_closest_flow, lucas_kanade_optical_flow, ICA_optical_flow
from handheld_super_resolution.robustness import compute_robustness
from handheld_super_resolution.kernels import estimate_kernels
from handheld_super_resolution.params import get_params
from handheld_super_resolution.merge import merge
from plot_flow import flow2img


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
                # translation = np.random.choice(np.linspace(-3, 3, 31), 2)
                # translation[1] *= 0


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
    blockspergrid_y = int(np.ceil(imsize[0]/threadsperblock[2]))
    blockspergrid = (blockspergrid_n, blockspergrid_x, blockspergrid_y)
    
    get_flow[blockspergrid, threadsperblock](upscaled_alignment, cuda_alignment)
    return upscaled_alignment.copy_to_host()

def align_bm(dec_burst, params):
    """returns tile wise BM alignment to the coarse scale"""
    grey_method_bm = params['block matching']['grey method']
    bayer_mode = params['mode']=='bayer'
    
    if bayer_mode :
        ref_grey, comp_grey = compute_grey_images(dec_burst[0], dec_burst[1:], grey_method_bm)
    else:
        ref_grey, comp_grey = dec_burst[0], dec_burst[1:]

    pre_alignment, _ = alignBurst(ref_grey, comp_grey,params['block matching'], options)
    pre_alignment = pre_alignment[:, :, :, ::-1] # swapping x and y direction (x must be first)
    if grey_method_bm in ["gauss", "decimating"] and bayer_mode:
        pre_alignment*=2
        # BM is always in grey mode, so input scale is output scale.
        # we choose by convention, to always return flow on the coarse scale, so x2 if grey is 2x smaller
    return pre_alignment
    
def align_lk(dec_burst, params, pre_alignment, mode="LK"):
    # warning this does not support grey mode, only bayer
    grey_method_lk = params['kanade']['grey method']
    options = {'verbose' : 3}
    ref_grey, comp_grey = compute_grey_images(dec_burst[0], dec_burst[1:], grey_method_lk)

    tile_size = params["kanade"]['tuning']['tileSize']
    if mode=='LK':
        lk_alignment = lucas_kanade_optical_flow(
            ref_grey, comp_grey, pre_alignment, options, params['kanade'], debug=True)
    elif mode =="ICA":
        lk_alignment = ICA_optical_flow(
            ref_grey, comp_grey, pre_alignment, options, params['kanade'], debug=True)
    else: 
        raise ValueError("invalid mode : {}".format(mode))
    if params["kanade"]['grey method'] in ['gauss', 'decimating']:
        for i in range(len(lk_alignment) - 1):
            lk_alignment[i] *=2 # last term has already been multiplied
            # during the last iteration
    

    lk_alignment.insert(0, pre_alignment)
    imsize = (dec_burst.shape[1], dec_burst.shape[2])
    
    upscaled = np.empty((len(lk_alignment), dec_burst.shape[0]-1, dec_burst.shape[1], dec_burst.shape[2], 2))
    for i in range(len(lk_alignment)): # x2 because coordinates are on bayer scale
        if grey_method_lk in ['FFT', 'demosaicing']:
            upscaled[i] = upscale_alignement(np.array(lk_alignment[i]), imsize, tile_size)
        else:
            upscaled[i] = upscale_alignement(np.array(lk_alignment[i]), imsize, tile_size*2)
    # we need to upscale because estimated_al is patchwise

    return lk_alignment, upscaled

def align_fb(dec_burst, params):
    grey_method_fb = 'FFT'
    ref_grey, comp_grey = compute_grey_images(dec_burst[0], dec_burst[1:], grey_method_fb)
        
    # Optical flow is now calculated
    farnback_flow = np.empty(comp_grey.shape+(2,))
    for i in range(farnback_flow.shape[0]):
        farnback_flow[i] = cv2.calcOpticalFlowFarneback(ref_grey, comp_grey[i], None, 0.5, 3, 16, 3, 5, 1.2, 0)
    upscaled_fb = np.empty( (dec_burst.shape[0] - 1, ) + dec_burst.shape[1:] + (2, ))
    # TODO upscale if methode is gauss or decimating
    upscaled_fb = farnback_flow
    if grey_method_fb in ['gauss', 'decimating']:
        return upscaled_fb*2 # greys are twice smaller
    else:
        return upscaled_fb

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


def evaluate_alignment(comp_alignment, comp_imgs, ref_img, gt_flow, label="", imshow=False, params=None):
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

    # warped_images = np.empty(comp_alignment.shape[:-1]+(3,))
    
    # im_EQ = np.empty(comp_alignment.shape[:-1])
    mean_flow_qe = np.empty(comp_alignment.shape[0])
    print("Evaluating {}".format(label))
    for iteration in  tqdm(range(comp_alignment.shape[0])):
    #     for image_index in range(comp_alignment.shape[1]):
    #         warped_images[iteration, image_index] = warp_flow(comp_imgs[image_index],
    #                                                           comp_alignment[iteration, image_index], rgb=True)
    #         im_EQ[iteration, image_index] = im_SE(ref_img,
    #                                               warped_images[iteration, image_index])
    #     # [it, image, posy, posx, flowxy
    #     # gt flow : image, flowxyu -> it=None, image, poxy=None, posx=None, flowy
        mean_flow_qe[iteration] = np.mean(np.linalg.norm(gt_flow[None][None][None].transpose((0,3,1,2,4)) - comp_alignment[iteration], axis=4), axis=(1,2,3))
    
    # last_im_MSE = np.mean(im_EQ[-1])
    if comp_alignment.shape[0] > 1:
        # plt.figure("flow MSE")
        # plt.plot([i for i in range(len(flow_EQ))], np.mean(flow_EQ, axis=(1,2,3)), label=label)
        # plt.xlabel('lk iteration')
        # plt.ylabel('MSE on flow')
        # plt.legend()
        
        # plt.figure("image MSE")
        # plt.plot([i for i in range(len(im_EQ))], np.mean(im_EQ, axis=(1,2,3)), label=label)
        # plt.xlabel('lk iteration')
        # plt.ylabel('MSE on warped image')
        # plt.legend()
    
        plt.figure("flow norm")
        plt.plot([np.mean(np.linalg.norm(comp_alignment[i], axis=3)) for i in range(comp_alignment.shape[0])], label=label)
        plt.xlabel('lk iteration')
        plt.ylabel('mean norm of optical flow')
        plt.legend()
        
        plt.figure("quadratic error on flow")
        plt.plot(mean_flow_qe, label=label)
        plt.xlabel('lk iteration')
        plt.ylabel("quadratic error on flow")
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
        
        # plt.figure("image MSE")
        # plt.plot([params['kanade']['tuning']['kanadeIter']], [last_im_MSE], marker = 'x', label = "Farneback")
        # plt.legend()
        
        plt.figure("flow norm")
        plt.plot([params['kanade']['tuning']['kanadeIter']], [np.mean(np.linalg.norm(comp_alignment[0], axis=3))] , marker = 'x', label = "Farneback")
        plt.xlabel('lk iteration')
        plt.ylabel('mean norm of optical flow')
        plt.legend()
        
        plt.figure("quadratic error on flow")
        plt.plot([params['kanade']['tuning']['kanadeIter']], [mean_flow_qe[0]], marker = 'x', label = "Farneback")
        plt.xlabel('lk iteration')
        plt.ylabel("quadratic error on flow")
        plt.legend()
        
    # if imshow : 
    #     for i in range(im_EQ.shape[0]):
    #         # plt.figure("{} alignment, step {}".format(label, i))
    #         # plt.imshow(np.log10(np.mean(flow_EQ[i], axis = 0)), cmap = "Reds")
    #         # plt.colorbar()
            
    #         plt.figure("{} warped alignment, step {}".format(label, i))
    #         plt.imshow(np.log10(im_EQ[i,1]),vmin = -3, vmax=4, cmap = "Reds")
    #         plt.colorbar()

    return mean_flow_qe

#%% params
#Warning : tileSize is expressed in terms of grey pixels.
CFA = np.array([[2, 1], [1, 0]])


params = get_params(PSNR=35)

###### Change paramaters here

params['block matching']['tuning']['factors'] = [1, 2, 2, 2] # a bit smaller because div 2k is not 4k
params['block matching']['grey method'] = "gauss"
params['kanade']['grey method'] = "gauss"
params['kanade']['tuning']['kanadeIter'] = 15
params['kanade']['tuning']['sigma blur'] = 1
params['robustness']['on'] = False

################################
params['robustness']['exif'] = {}
params['merging']['exif'] = {}
params['merging']['exif']['CFA Pattern'] = CFA
params['robustness']['exif']['CFA Pattern'] = CFA

# copying parameters values in sub-dictionaries
if 'scale' not in params["merging"].keys() :
    params["merging"]["scale"] = params["scale"]
if 'tileSize' not in params["kanade"]["tuning"].keys():
    params["kanade"]["tuning"]['tileSize'] = params['block matching']['tuning']['tileSizes'][0]
if 'tileSize' not in params["robustness"]["tuning"].keys():
    params["robustness"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']
if 'tileSize' not in params["merging"]["tuning"].keys():
    params["merging"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']

# if 'mode' not in params["block matching"].keys():
#     params["block matching"]["mode"] = params['mode']
if 'mode' not in params["kanade"].keys():
    params["kanade"]["mode"] = params['mode']
if 'mode' not in params["robustness"].keys():
    params["robustness"]["mode"] = params['mode']
if 'mode' not in params["merging"].keys():
    params["merging"]["mode"] = params['mode']

# systematically grey, so we can control internally how grey is obtained
params["block matching"]["mode"] = 'grey'
if params["block matching"]["grey method"] in ["FFT", "demosaicing"]:
    params["block matching"]['tuning']["tileSizes"] = [ts*2 for ts in params["block matching"]['tuning']["tileSizes"]]
if params["kanade"]["grey method"] in ["FFT", "demosaicing"]:
    params["kanade"]['tuning']["tileSize"] *= 2


params['robustness']['std_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_std_ISO_50.npy')
params['robustness']['diff_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_diff_ISO_50.npy')
options = {'verbose' : 3}

small_tileSizes = params["block matching"]['tuning']["tileSizes"].copy()
small_Ts = small_tileSizes[0]

big_tileSizes = [ts*2 for ts in small_tileSizes]
big_Ts = small_Ts*2

#%% generating burst
if __name__=="__main__":
    # img = plt.imread("P:/Kodak/1.png")*255
    # img = plt.imread("P:/mire.png")[:,:,:3]*255
    # img = plt.imread("P:/DIV2K_valid_HR/DIV2K_valid_HR/0820.png")*255
    #img = plt.imread("P:/Urban100_SR/image_SRF_4/img_040_SRF_4_HR.png")*255
    # img = plt.imread("P:/0002/Canon/im.JPG")
    img = plt.imread("P:/DIV2K_valid_HR/DIV2K_valid_HR/0900.png")*255
    transformation_params = {'max_translation':5,
                              'max_shear': 0,
                              'max_ar_factor': 0,
                              'max_rotation': 0}
    burst, flow = single2lrburst(img, 6, downsample_factor=2, transformation_params=transformation_params)
    # flow is unussable because it is pointing from moving frame to ref. We would need the opposite
    
    
    dec_burst = (decimate(burst)/255).astype(np.float32)
    
    grey_burst = np.mean(burst, axis = 3)/255

#%% testing pipleine on one bayer image
    params["block matching"]["mode"] = 'bayer'
    params["kanade"]["mode"] = 'bayer'
    params["merging"]["mode"] = 'bayer'
    params["robustness"]["mode"] = 'bayer'
    
    
    output, R, r, alignment, covs = main(dec_burst[0], dec_burst[1:], options, params)
    plt.figure("merge on bayer images new")
    plt.imshow(output[:,:,:3])
    plt.figure("ref")
    plt.imshow(cv2.resize(colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(dec_burst[0], pattern='BGGR'), None, fx = params["merging"]['scale'], fy = params["merging"]['scale'], interpolation=cv2.INTER_CUBIC))

#%% aligning LK on bayer
    df = pd.DataFrame()

    ground_truth_flow = flow[1:,:,0,0]
    
    t1 = time()
    fb_alignment = align_fb(dec_burst*255, params)
    print('farneback evaluated : ', time()-t1)
    eq = evaluate_alignment(fb_alignment[None], burst[1:]/255, burst[0]/255, ground_truth_flow, label = "FarneBack", imshow=True, params=params)

    
    ## FFT BM
    params["block matching"]["grey method"] = "FFT"
    params["block matching"]['tuning']["tileSizes"] = big_tileSizes
    pre_alignment = align_bm(dec_burst, params)
    
    
    # mode = "LK"
    # params["kanade"]["grey method"] = "FFT"
    # params["kanade"]['tuning']['tileSize'] = big_Ts
    # label = "BM {}, {} {}, subpixels {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"], params['block matching']['tuning']['subpixels'])
    # raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    # lk_warped_images, lk_im_EQ = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)

    mode = "ICA"
    params["kanade"]["grey method"] = "FFT"
    params["kanade"]['tuning']['tileSize'] = big_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)
    df[label] = eq
    
    params["kanade"]["grey method"] = "decimating"
    params["kanade"]['tuning']['tileSize'] = small_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow,  label = label, imshow=False, params=params)
    df[label] = eq

    # params["kanade"]["grey method"] = "gauss"
    # params["kanade"]['tuning']['tileSize'] = small_Ts
    # label = "BM {}, LK {}".format(params["block matching"]["grey method"],  params["kanade"]["grey method"])
    # raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment)
    # lk_warped_images, lk_im_EQ = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow,  label = label, imshow=False, params=params)

    params["kanade"]["grey method"] = "demosaicing"
    params["kanade"]['tuning']['tileSize'] = big_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)
    df[label] = eq
    
    # demosaicing BM
    params["block matching"]["grey method"] = "demosaicing"
    params["block matching"]['tuning']["tileSizes"] = big_tileSizes
    pre_alignment = align_bm(dec_burst, params)
    
    params["kanade"]["grey method"] = "decimating"
    params["kanade"]['tuning']['tileSize'] = small_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow,  label = label, imshow=False, params=params)
    df[label] = eq
    
    params["kanade"]["grey method"] = "FFT"
    params["kanade"]['tuning']['tileSize'] = big_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)
    df[label] = eq
    
    params["kanade"]["grey method"] = "demosaicing"
    params["kanade"]['tuning']['tileSize'] = big_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)
    df[label] = eq

    # gauss BM
    params["block matching"]["grey method"] = "gauss"
    params["block matching"]['tuning']["tileSizes"] = small_tileSizes
    pre_alignment = align_bm(dec_burst, params)
    
    params["kanade"]["grey method"] = "decimating"
    params["kanade"]['tuning']['tileSize'] = small_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow,  label = label, imshow=False, params=params)
    df[label] = eq
    
    params["kanade"]["grey method"] = "FFT"
    params["kanade"]['tuning']['tileSize'] = big_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)
    df[label] = eq

    params["kanade"]["grey method"] = "demosaicing"
    params["kanade"]['tuning']['tileSize'] = big_Ts
    label = "BM {}, {} {}".format(params["block matching"]["grey method"], mode, params["kanade"]["grey method"])
    raw_lk_alignment, upscaled_lk_alignment = align_lk(dec_burst, params, pre_alignment, mode)
    eq = evaluate_alignment(upscaled_lk_alignment, burst[1:]/255, burst[0]/255, ground_truth_flow, label = label, imshow=False, params=params)
    df[label] = eq
    
#%% plot flow
    maxrad = 10
    
    for image_index in range(dec_burst.shape[0]-1):
        flow_image = flow2img(upscaled_lk_alignment[-1, image_index], maxrad)
        plt.figure("flow FFT {}".format(image_index))
        plt.imshow(flow_image)
        if image_index==1 : 
            break
    
    # Making color wheel
    X = np.linspace(-maxrad, maxrad, 1000)
    Y = np.linspace(-maxrad, maxrad, 1000)
    Xm, Ym = np.meshgrid(X, Y)
    Z = np.stack((Xm, Ym)).transpose(1,2,0)
    # Z = np.stack((-Z[:,:,1], -Z[:,:,0]), axis=-1)
    flow_image = flow2img(Z, maxrad)
    plt.figure("wheel")
    plt.imshow(flow_image)

#%% ploting burst
    plt.figure("ref")
    plt.imshow(burst[0]/255)
    for i in range(4):
        plt.figure("{}".format(i))
        plt.imshow(burst[i+1]/255)

#%% matching warps with original

    plt.figure("ref")
    plt.imshow(burst[0]/255)
    for i in range(fb_warped_images.shape[1]):
        plt.figure("{}".format(i))
        plt.imshow(lk_warped_images[-1, i])


#%%
    plt.figure("LK")
    plt.imshow(lk_warped_images[-1, 0])
    plt.figure("Farneback")
    plt.imshow(fb_warped_images[0, 0])
    plt.figure("Block Matching")
    plt.imshow(lk_warped_images[0,0])



#%% color accumulation check

    # ref_img = plt.imread('P:/mire.png')*255
    ref_img = plt.imread('P:/DIV2K_valid_HR/DIV2K_valid_HR/0820.png')*255
    comp_imgs = np.zeros((4, ) + ref_img.shape)
    
    comp_imgs[0] = ref_img
    comp_imgs[1, :, 1:] = ref_img[:,:-1]
    comp_imgs[2, 1:, :] = ref_img[:-1,:]
    comp_imgs[3, 1:, 1:] = ref_img[:-1,:-1]
    
    dec_burst = decimate(comp_imgs)/255


    params["block matching"]["mode"] = 'bayer'
    params["kanade"]["mode"] = 'bayer'
    params["merging"]["mode"] = 'bayer'
    params["robustness"]["mode"] = 'bayer'
    
    output, R, r, alignment, kernels = main(dec_burst[0].astype(np.float32), dec_burst[1:].astype(np.float32), options, params)
    plt.figure("merge on bayer images")
    plt.imshow(output[:,:,:3])
    plt.figure("ref")
    plt.imshow(cv2.resize(colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(dec_burst[0], pattern='BGGR'), None, fx = params["merging"]['scale'], fy = params["merging"]['scale'], interpolation=cv2.INTER_CUBIC))


#%% pipeline with ground truth flow

    tile_size = params['kanade']['tuning']['tileSize']
    n_images, imsize_y, imsize_x = dec_burst.shape
    n_patch_y = 2 * math.ceil(imsize_y/tile_size) + 1 
    n_patch_x = 2 * math.ceil(imsize_x/tile_size) + 1
    
    ground_truth_flow = np.empty((n_images-1, n_patch_y, n_patch_x, 2))
    for i in range(n_images-1):
        ground_truth_flow[i, :, :, 0] = flow[i+1, 0, 0, 0]
        ground_truth_flow[i, :, :, 1] = flow[i+1, 1, 0, 0]
    cuda_final_alignment = cuda.to_device(ground_truth_flow)
    
    cuda_ref_img = cuda.to_device(dec_burst[0])
    cuda_comp_imgs = cuda.to_device(dec_burst[1:])
    
    
    cuda_Robustness, cuda_robustness = compute_robustness(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment,
                                              options, params['robustness'])
    
    cuda_kernels = estimate_kernels(dec_burst[0], dec_burst[1:], options, params['merging'])  
    covs = cuda_kernels.copy_to_host()
    output = merge(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment, cuda_kernels, cuda_robustness, options, params['merging'])

#%% ploting
    # postprocessed_output = output[:,:,:3]
    postprocessed_output = filters.unsharp_mask(output[:,:,:3], radius=3, amount=1,
                                channel_axis=2, preserve_range=True)
    postprocessed_bicubic = cv2.resize(colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(dec_burst[0], pattern='BGGR'), None, fx = params["merging"]['scale'], fy = params["merging"]['scale'], interpolation=cv2.INTER_CUBIC)
    
    
    
    
    plt.figure("merge on bayer images, kernel {} aniso".format(params['merging']['kernel']))
    plt.imshow(postprocessed_output)
    plt.scatter(350, 350, marker='x', c='r')
    plt.figure("ref bicubic")
    plt.imshow(postprocessed_bicubic)
    plt.scatter(350, 350, marker='x', c='r')
    plt.figure("ref ")
    plt.imshow(colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(dec_burst[0], pattern='BGGR'))
    plt.scatter(350/params["merging"]['scale'], 350/params["merging"]['scale'],
                marker='x', c='r')
    
    
    # plotting crops
    crops_x = 2
    crops_y = 2
    crop_size = (int(output.shape[0]/crops_y), int(output.shape[1]/crops_x))
    
    for i in range(crops_y):
        for j in range(crops_x):
            if (i == 0 and j == 1) or (i == 1 and j == 1):
                plt.figure("output {} {}, kernel {} aniso".format(i,j, params['merging']['kernel']))
                plt.imshow(postprocessed_output[i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]])
                plt.scatter(crop_size[0]/2, crop_size[1]/2, marker='x', c='r')
                
                plt.figure("original bicubic {} {}".format(i,j))
                plt.imshow(postprocessed_bicubic[i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]])
                plt.scatter(crop_size[0]/2, crop_size[1]/2, marker='x', c='r')

#%% plotting merge

    # D : Dx, Dy, bayer pixel (0, 1, 2 or 3). DIstance to the central pixel of the 3x3 patch,
    # and associated channel
    D = np.empty((dec_burst.shape[0]+1, output.shape[0], output.shape[1], 3))
    for image in tqdm(range(dec_burst.shape[0]+1)):
        D[image, :, :,0] = output[:,:,3 + 3*image]
        D[image, :, :,1] = output[:,:,3 + 3*image + 1]
        D[image, :, :,2] = output[:,:,3 + 3*image + 2]
            
    
    def plot_merge(covs, Dist, pos):
        bayer= CFA
        reframed_posx, x = math.modf(pos[1]/(2*params['scale'])) # these positions are between 0 and 1
        reframed_posy, y = math.modf(pos[0]/(2*params['scale']))
        x=int(x); y=int(y)
        
        int_cov = (covs[0, y, x]*(1 - reframed_posx)*(1 - reframed_posy) +
                   covs[0, y, x+1]*(reframed_posx)*(1 - reframed_posy) + 
                   covs[0, y+1, x]*(1 - reframed_posx)*(reframed_posy) + 
                   covs[0, y+1, y+1]*reframed_posx*reframed_posy)
        
        cov_i = np.linalg.inv(int_cov*params['scale']**2)
    
        
        # plotting kernel
        L = np.linspace(-5, 5, 100)
        Xm, Ym = np.meshgrid(L,L)
        Z = np.empty_like(Xm)
        Z = cov_i[0,0] * Xm**2 + (cov_i[1, 0] + cov_i[0, 1])*Xm*Ym + cov_i[1, 1]*Ym**2
        plt.figure("merge in (x = {}, y = {})".format(pos[1],pos[0]))
        plt.pcolor(Xm, Ym, np.exp(-Z/2), vmin = 0, vmax=1)
        plt.gca().invert_yaxis()
        plt.scatter([0], [0], c='k', marker ='o')
        
        
        
        colors = ['r', 'g', 'b']
        
        # scattering frames pixels.
        for image in range(Dist.shape[0]):
            D = Dist[(image,) + pos]
            x_center, y_center, channel = D
            # [0, 1, 2, 3] -> [0, 1]x[0, 1]
            bayer_idy_center, bayer_idx_center = channel//2, channel%2 
            
            for idx in range(-1,2):
                for idy in range(-1, 2):
                    x = x_center + idx*params["scale"]
                    y = y_center + idy*params["scale"]

                    bayer_idx =int((bayer_idx_center + idx)%2)
                    bayer_idy = int((bayer_idy_center + idy)%2)
                    c = colors[CFA[bayer_idy, bayer_idx]]
                    plt.scatter(x,y,c=c,marker='x')            

        
    
        plt.colorbar()