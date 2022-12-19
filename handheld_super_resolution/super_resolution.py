# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

@author: jamyl
"""

import os
import glob
import time

from pathlib import Path
import numpy as np
from numba import cuda
import exifread
import rawpy


from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, crop, divide
from .utils_image import compute_grey_images
from .merge import merge, init_merge
from .kernels import estimate_kernels
from .block_matching import init_block_matching, align_image_block_matching
from .optical_flow import ICA_optical_flow, init_ICA
from .robustness import init_robustness, compute_robustness
from .params import check_params_validity

NOISE_MODEL_PATH = Path(os.getcwd()) / 'data' 
        


def main(ref_img, comp_imgs, options, params):
    verbose = options['verbose'] >= 1
    verbose_2 = options['verbose'] >= 2
    verbose_3 = options['verbose'] >= 3
    
    bayer_mode = params['mode']=='bayer'
    
    debug_mode = params['debug']
    if debug_mode : 
        debug_dict = {"robustness":[],
                      "flow":[]}

    if verbose :
        cuda.synchronize()
        print("\nProcessing reference image ---------\n")
        im_time = time.perf_counter()

    #___ Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    
    
    #___ Raw to grey
    grey_method = params['grey method']
    t1 = time.perf_counter()
    
    if bayer_mode :
        cuda_ref_grey = compute_grey_images(cuda_ref_img, grey_method)
        if verbose_3 :
            getTime(t1, "- Ref grey image estimated by {}".format(grey_method))
    else:
        cuda_ref_grey = cuda_ref_img
        
    #___ Block Matching
    referencePyramid = init_block_matching(cuda_ref_grey, options, params['block matching'])
        
    
    #___ ICA : compute grad and hessian
    ref_gradx, ref_grady, hessian = init_ICA(cuda_ref_grey, options, params['kanade'])
    
    
    #___ Local stats estimation
    ref_local_stats = init_robustness(cuda_ref_img,options, params['robustness'])

        
    #___ Kernel estimation
    if verbose_2 : 
        current_time = time.perf_counter()
        print('Estimating kernels')
    cuda_kernels = estimate_kernels(cuda_ref_img, options, params['merging'])
    if verbose_2 : 
        current_time = getTime(
            current_time, 'Kernels estimated (Total)')
    
    
    #___ init merge
    num, den = init_merge(cuda_ref_img, cuda_kernels, options, params["merging"])
    
    if verbose :
        cuda.synchronize()
        getTime(im_time, 'Image processed (Total)')
    

    n_images = comp_imgs.shape[0]
    for im_id in range(n_images):
        if verbose :
            cuda.synchronize()
            print("\nProcessing image {} ---------\n".format(im_id+1))
            im_time = time.perf_counter()
        
        #___ Moving to GPU
        cuda_img = cuda.to_device(comp_imgs[im_id])
        if verbose_3 : 
            cuda.synchronize()
            current_time = getTime(
                im_time, 'Arrays moved to GPU')
        
        #___ Compute Grey Images
        if bayer_mode:
            cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)
            if verbose_3 :
                cuda.synchronize()
                current_time = getTime(current_time, "- grey images estimated by {}".format(grey_method))
        else:
            cuda_im_grey = cuda_img
        
        #___ Block Matching
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('Beginning block matching')
        
        pre_alignment = align_image_block_matching(cuda_im_grey, referencePyramid, options, params['block matching'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Block Matching (Total)')
        #___ ICA
        
        cuda_final_alignment = ICA_optical_flow(
            cuda_im_grey, cuda_ref_grey, ref_gradx, ref_grady, hessian, pre_alignment, options, params['kanade'])
        
        #___ Robustness
        if verbose_2 : 
            cuda.synchronize()
            current_time = time.perf_counter()
            
        cuda_robustness = compute_robustness(cuda_img, ref_local_stats, cuda_final_alignment,
                                             options, params['robustness'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, 'Robustness estimated (Total)')
            print('\nEstimating kernels')
            
        #___ Kernel estimation
        cuda_kernels = estimate_kernels(cuda_img, options, params['merging'])
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(
                current_time, 'Kernels estimated (Total)')
            
        #___ Merging
        merge(cuda_img, cuda_final_alignment, cuda_kernels, cuda_robustness, num, den,
              options, params['merging'])
        
        if verbose :
            cuda.synchronize()
            getTime(im_time, 'Image processed (Total)')
            
        if debug_mode : 
            debug_dict['robustness'].append(cuda_robustness.copy_to_host())
    
    if verbose_2 :
        cuda.synchronize()
        current_time = time.perf_counter()
        
    # num is outwritten into num/den
    divide(num, den)
    
    if verbose_2 :
        cuda.synchronize()
        current_time = getTime(
            current_time, 'Image normalized (Total)')
        
    output = num.copy_to_host()
    if verbose :
        print('\nTotal ellapsed time : ', time.perf_counter() - t1)
        
    if debug_mode :
        return output, debug_dict
    else:
        return output

#%%

def process(burst_path, options, params, crop_str=None):
    """
    Process the burst

    Parameters
    ----------
    burst_path : str
        Path where the .dng burst is located
    options : dict
        
    params : Parameters
        See params.py for more details.
    crop_str : str, optional
        A crop that should be applied before processing the images.
        It must have an even shift in every direction, so that the CFA remains the same.
        The default is None.
        
        Example : "[500:2500, 1000:2500]"

    Returns
    -------
    Array
        The processed image

    """
    currentTime, verbose = time.perf_counter(), options['verbose'] > 2
    
    ref_id = 0 #TODO Select ref id based on HDR+ method
    
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
    
    # checking parameters coherence
    check_params_validity(params)
    
    # Cropping the image if needed
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
    
    black_levels = tags['Image BlackLevel'] # This tag is a fraction object for some reason. It seems that black levels are all integers anyway
    black_levels = np.array([int(x.decimal()) for x in black_levels.values])
    
    white_balance = raw.camera_whitebalance
    
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


    
    if np.issubdtype(type(ref_raw[0,0]), np.integer):
        ## Here do black and white level correction and white balance processing for all image in comp_images
        ## Each image in comp_images should be between 0 and 1.
        ## ref_raw is a (H,W) array
        ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        for i in range(2):
            for j in range(2):
                channel = CFA[i, j]
                ref_raw[i::2, j::2] = (ref_raw[i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
                ref_raw[i::2, j::2] *= white_balance[channel] / white_balance[1]
        
        
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
                raw_comp[:, i::2, j::2] *= white_balance[channel] / white_balance[1]
        raw_comp = np.clip(raw_comp, 0., 1.)

    return main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), options, params)
