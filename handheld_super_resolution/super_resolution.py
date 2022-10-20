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
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
import exifread
import rawpy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, crop
from .merge import merge
from .block_matching import alignBurst
from .optical_flow import lucas_kanade_optical_flow
from .robustness import compute_robustness

NOISE_MODEL_PATH = Path(os.getcwd()) / 'data' 

def main(ref_img, comp_imgs, options, params):
    verbose = options['verbose'] > 1
    verbose_2 = options['verbose'] > 2
    
    
    t1 = time()
    if verbose :
        print('Beginning block matching')
    pre_alignment, _ = alignBurst(ref_img, comp_imgs,params['block matching'], options)
    pre_alignment = pre_alignment[:, :, :, ::-1] # swapping x and y direction (x must be first)
    
    if verbose : 
        current_time = getTime(t1, 'Block Matching (Total)')
    
    
    cuda_ref_img = cuda.to_device(ref_img)
    cuda_comp_imgs = cuda.to_device(comp_imgs)
    
    if verbose_2 : 
        current_time = getTime(
            current_time, 'Arrays moved to GPU')
    
    
    cuda_final_alignment = lucas_kanade_optical_flow(
        ref_img, comp_imgs, pre_alignment, options, params['kanade'])
    
    
    if verbose : 
        current_time = time()

    
    cuda_Robustness, cuda_robustness = compute_robustness(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment,
                                             options, params['robustness'])
    if verbose : 
        current_time = getTime(
            current_time, 'Robustness estimated (Total)')
    
    output = merge(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment, cuda_robustness, options, params['merging'])
    if verbose:
        print('\nTotal ellapsed time : ', time() - t1)
    return output, cuda_Robustness.copy_to_host(), cuda_robustness.copy_to_host(), cuda_final_alignment.copy_to_host()

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

    if 'mode' not in params["block matching"].keys():
        params["block matching"]["mode"] = params['mode']
    if 'mode' not in params["kanade"].keys():
        params["kanade"]["mode"] = params['mode']
    if 'mode' not in params["robustness"].keys():
        params["robustness"]["mode"] = params['mode']
    if 'mode' not in params["merging"].keys():
        params["merging"]["mode"] = params['mode']
    
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
