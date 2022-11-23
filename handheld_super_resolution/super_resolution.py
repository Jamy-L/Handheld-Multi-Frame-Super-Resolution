# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

@author: jamyl
"""

import os
import glob
from time import time

from pathlib import Path
import numpy as np
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32, typeof
import exifread
import rawpy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft2, ifft2, fftshift, ifftshift

import colour_demosaicing
from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, crop
from .utils_image import downsample, compute_grey_images
from .merge import merge, init_merge, division
from .kernels import estimate_kernels
from .block_matching import alignBurst, init_block_matching, align_image_block_matching
from .optical_flow import lucas_kanade_optical_flow, ICA_optical_flow, init_ICA
from .robustness import init_robustness, compute_robustness
from .params import check_params_validity

NOISE_MODEL_PATH = Path(os.getcwd()) / 'data' 
        
def main(ref_img, comp_imgs, options, params):
    verbose = options['verbose'] > 1
    verbose_2 = options['verbose'] > 2
    bayer_mode = params['mode']=='bayer'
    
    
    #___ Raw to grey
    grey_method = params['grey method']
    
    if bayer_mode :
        t1 = time()
        ref_grey = compute_grey_images(ref_img, grey_method)
        if verbose_2 :
            currentTime = getTime(t1, "- Ref grey image estimated by {}".format(grey_method))
    else:
        ref_grey = ref_img
        
    #___ Block Matching
    
    referencePyramid = init_block_matching(ref_grey, options, params['block matching'])

    
    #___ Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    cuda_ref_grey = cuda.to_device(ref_grey)
    
    
    #___ ICA : compute grad and hessian
    ref_gradx, ref_grady, hessian = init_ICA(ref_grey, options, params['kanade'])
    
    
    #___ Local stats estimation
    ref_local_stats = init_robustness(cuda_ref_img,options, params['robustness'])

        
    #___ Kernel estimation
    if verbose : 
        current_time = time()
        print('Estimating kernels')
    cuda_kernels = estimate_kernels(ref_img, options, params['merging'])
    if verbose : 
        current_time = getTime(
            current_time, 'Kernels estimated (Total)')
    
    
    #___ init merge
    num, den = init_merge(cuda_ref_img, cuda_kernels, options, params["merging"])

    
    n_images = 8 #comp_imgs.shape[0]
    for im_id in range(n_images):
        if verbose :
            print("\nProcessing image {} ---------\n".format(im_id+1))
        im_time = time()
        
        if bayer_mode:
            im_grey = compute_grey_images(comp_imgs[im_id], grey_method)
            if verbose_2 :
                current_time = getTime(im_time, "- LK grey images estimated by {}".format(grey_method))
        else:
            im_grey = comp_imgs[im_id]
        
        #___ Block Matching
        current_time = time()
        if verbose :
            print('Beginning block matching')
        
        pre_alignment = align_image_block_matching(im_grey, referencePyramid, options, params['block matching'])
        
        if verbose : 
            current_time = getTime(current_time, 'Block Matching (Total)')
        #___ Moving to GPU
        cuda_img = cuda.to_device(comp_imgs[im_id])
        cuda_im_grey = cuda.to_device(im_grey)
        
        if verbose_2 : 
            current_time = getTime(
                current_time, 'Arrays moved to GPU')
            
        cuda_final_alignment = ICA_optical_flow(
            cuda_im_grey, cuda_ref_grey, ref_gradx, ref_grady, hessian, pre_alignment, options, params['kanade'])
    
        #___ Robustness
        cuda.synchronize()
        if verbose : 
            current_time = time()
        cuda_Robustness, cuda_robustness = compute_robustness(cuda_img, ref_local_stats, cuda_final_alignment,
                                                 options, params['robustness'])
        if verbose : 
            current_time = getTime(
                current_time, 'Robustness estimated (Total)')
            print('\nEstimating kernels')
            
        #___ Kernel estimation
        cuda_kernels = estimate_kernels(comp_imgs[im_id], options, params['merging'])
        cuda.synchronize()
        if verbose : 
            current_time = getTime(
                current_time, 'Kernels estimated (Total)')
            
        #___ Merging
        merge(cuda_img, cuda_final_alignment, cuda_kernels, cuda_robustness, num, den,
              options, params['merging'])
        if verbose :
            getTime(
                im_time, 'Image processed (Total)')
    
    # num is outwritten into num/den
    channels = num.shape[-1]
    threadsperblock = (16, 16, channels) # may be modified (3 h)
    blockspergrid_x = int(np.ceil(num.shape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(num.shape[0]/threadsperblock[0]))
    blockspergrid_z = 1

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    division[blockspergrid, threadsperblock](num, den)
    
    output = num.copy_to_host()

    if verbose : 
        current_time = getTime(
            current_time, 'Merge finished (Total)')
        print('\nTotal ellapsed time : ', time() - t1)
        
    return output, cuda_final_alignment

#%%

def process(burst_path, options, params, crop_str=None):
    currentTime, verbose = time(), options['verbose'] > 2
    
    
    ref_id = 0 #TODO get ref id
    
    raw_comp = []
    
    # Get the list of raw images in the burst path
    raw_path_list = glob.glob(os.path.join(burst_path, '*.dng'))
    assert raw_path_list != [], 'At least one raw .dng file must be present in the burst folder.'
	# Read the raw bayer data from the DNG files
    for index, raw_path in enumerate(raw_path_list):
        with rawpy.imread(raw_path) as rawObject:
            if index != ref_id :
                
                raw_comp.append(rawObject.raw_image.copy())  # copy otherwise image data is lost when the rawpy object is closed
    raw_comp = np.array(raw_comp)
    # Reference image selection and metadata     
    raw = rawpy.imread(raw_path_list[ref_id])
    ref_raw = raw.raw_image.copy()
    check_params_validity(params)
    
    if crop_str is not None:
        ref_raw = crop(ref_raw, crop_str, axis=(0, 1))
        raw_comp = crop(raw_comp, crop_str, axis=(1, 2))
    
    
    with open(raw_path_list[ref_id], 'rb') as raw_file:
        tags = exifread.process_file(raw_file)
    
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
        if not 'exif' in params['robustness'].keys(): 
            params['robustness']['exif'] = {}
        
    white_level = tags['Image Tag 0xC61D'].values[0] # there is only one white level
    
    black_levels = tags['Image BlackLevel'] # This tag is a fraction for some reason. It seems that black levels are all integers anyway
    black_levels = np.array([int(x.decimal()) for x in black_levels.values])
    
    white_balance = raw.camera_whitebalance # TODO make sure green is 1 and other ratio
    
    CFA = tags['Image CFAPattern']
    CFA = np.array([x for x in CFA.values]).reshape(2,2)
    
    params['merging']['exif']['CFA Pattern'] = CFA
    params['robustness']['exif']['CFA Pattern'] = CFA
    params['ISO'] = int(str(tags['Image ISOSpeedRatings']))
    
    # Packing noise model related to picture ISO
    if ('std_curve' not in params['robustness'].keys()) or \
        ('diff_curve' not in params['robustness'].keys()):
            
        std_noise_model_label = 'noise_model_std_ISO_{}'.format(params['ISO'])
        diff_noise_model_label = 'noise_model_diff_ISO_{}'.format(params['ISO'])
        std_noise_model_path = (NOISE_MODEL_PATH / std_noise_model_label).with_suffix('.npy')
        diff_noise_model_path = (NOISE_MODEL_PATH / diff_noise_model_label).with_suffix('.npy')
        
        params['robustness']['std_curve'] = np.load(std_noise_model_path)
        params['robustness']['diff_curve'] = np.load(diff_noise_model_path)
    
    
    if verbose:
        currentTime = getTime(currentTime, ' -- Read raw files')
        
    # copying parameters values in sub-dictionaries
    if 'scale' not in params["merging"].keys() :
        params["merging"]["scale"] = params["scale"]
    if 'tileSize' not in params["kanade"]["tuning"].keys():
        params["kanade"]["tuning"]['tileSize'] = params['block matching']['tuning']['tileSizes'][0]
    if 'tileSize' not in params["robustness"]["tuning"].keys():
        params["robustness"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']
    if 'tileSize' not in params["merging"]["tuning"].keys():
        params["merging"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']


    if 'mode' not in params["kanade"].keys():
        params["kanade"]["mode"] = params['mode']
    if 'mode' not in params["robustness"].keys():
        params["robustness"]["mode"] = params['mode']
    if 'mode' not in params["merging"].keys():
        params["merging"]["mode"] = params['mode']
        
    params['kanade']['grey method'] = params['grey method']
    
    # systematically grey, so we can control internally how grey is obtained
    params["block matching"]["mode"] = 'grey'
    if params["grey method"] in ["FFT", "demosaicing"]:
        params["block matching"]['tuning']["tileSizes"] = [ts*2 for ts in params["block matching"]['tuning']["tileSizes"]]
        params["kanade"]['tuning']["tileSize"] *= 2
        
    if params['mode'] == 'bayer':
        params["merging"]["tuning"]['tileSize'] *= 2

    
        

    
    if np.issubdtype(type(ref_raw[0,0]), np.integer):
        ## Here do black and white level correction and white balance processing for all image in comp_images
        ## Each image in comp_images should be between 0 and 1.
        ## ref_raw is a (H,W) array
        ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        for i in range(2):
            for j in range(2):
                channel = channel = CFA[i, j]
                ref_raw[i::2, j::2] = (ref_raw[i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
                ref_raw[i::2, j::2] *= white_balance[channel]
        
        
        # images[:, 0::2, 1::2] *= float(ref_raw.camera_whitebalance[0]) / raw.camera_whitebalance[1]
        # images[:, 1::2, 0::2] *= float(ref_raw.camera_whitebalance[2]) / raw.camera_whitebalance[1]
        ref_raw = np.clip(ref_raw, 0.0, 1.0)
        # ## The division by the green WB value is important because WB may come with integer coefficients instead
        
    if np.issubdtype(type(raw_comp[0,0,0]), np.integer):
        raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        ## raw_comp is a (N, H,W) array
        for i in range(2):
            for j in range(2):
                channel = channel = CFA[i, j]
                raw_comp[:, i::2, j::2] = (raw_comp[:, i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
                raw_comp[:, i::2, j::2] *= white_balance[channel]

    return main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), options, params)
