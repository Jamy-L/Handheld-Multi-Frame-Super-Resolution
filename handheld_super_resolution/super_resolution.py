# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

This script contains : 
    - The implementation of Alg. 1, the main the body of the method
    - The implementation of Alg. 2, where the function necessary to
        compute the optical flow are called
    - All the operations necessary before its call, such as whitebalance,
        exif reading, and manipulations of the user's parameters.
    - The call to post-processing operations (if enabled)


@author: jamyl
"""

import os
import time
import warnings

from pathlib import Path
import numpy as np
from numba import cuda
import rawpy

from .utils_image import compute_grey_images, frame_count_denoising_gauss, frame_count_denoising_median, apply_orientation
from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, divide, add, round_iso
from .block_matching import init_block_matching, align_image_block_matching
from .params import check_params_validity, get_params, merge_params
from .robustness import init_robustness, compute_robustness
from .utils_dng import load_dng_burst
from .ICA import ICA_optical_flow, init_ICA
from .fast_monte_carlo import run_fast_MC
from .kernels import estimate_kernels
from .merge import merge, merge_ref
from . import raw2rgb

NOISE_MODEL_PATH = Path(os.path.dirname(__file__)).parent / 'data' 
        

def main(ref_img, comp_imgs, options, params):
    """
    This is the implementation of Alg. 1: HandheldBurstSuperResolution.
    Some part of Alg. 2: Registration are also integrated for optimisation.

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Reference frame J_1
    comp_imgs : Array[N-1, imshape_y, imshape_x]
        Remaining frames of the burst J_2, ..., J_N
        
    options : dict
        verbose options.
    params : dict
        paramters.

    Returns
    -------
    num : device Array[imshape_y*s, imshape_y*s, 3]
        generated RGB image WITHOUT any post-processing.
    debug_dict : dict
        Contains (if debugging is enabled) some debugging infos.

    """
    verbose = options['verbose'] >= 1
    verbose_2 = options['verbose'] >= 2
    verbose_3 = options['verbose'] >= 3
    
    bayer_mode = params['mode']=='bayer'
    
    debug_mode = params['debug']
    debug_dict = {"robustness":[],
                  "flow":[]}
    
    accumulate_r = params['accumulated robustness denoiser']['on']

    #### Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    cuda.synchronize()
    
    if verbose :
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()
    
    
    #### Raw to grey
    grey_method = params['grey method']
    
    if bayer_mode :
        cuda_ref_grey = compute_grey_images(cuda_ref_img, grey_method)
        if verbose_3 :
            cuda.synchronize()
            getTime(t1, "- Ref grey image estimated by {}".format(grey_method))
    else:
        cuda_ref_grey = cuda_ref_img
        
    #### Block Matching
    if verbose_2 :
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nBeginning Block Matching initialisation')
        
    reference_pyramid = init_block_matching(cuda_ref_grey, options, params['block matching'])
    
    if verbose_2 :
        cuda.synchronize()
        current_time = getTime(current_time, 'Block Matching initialised (Total)')
        
    
    #### ICA : compute grad and hessian    
    if verbose_2 :
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nBeginning ICA initialisation')
        
    ref_gradx, ref_grady, hessian = init_ICA(cuda_ref_grey, options, params['kanade'])
    
    if verbose_2 :
        cuda.synchronize()
        current_time = getTime(current_time, 'ICA initialised (Total)')
    
    
    #### Local stats estimation
    if verbose_2:
        cuda.synchronize()
        current_time = time.perf_counter()
        print("\nEstimating ref image local stats")
        
    ref_local_means, ref_local_stds = init_robustness(cuda_ref_img,options, params['robustness'])
    
    if accumulate_r:
        accumulated_r = cuda.to_device(np.zeros(ref_local_means.shape[:2]))

    
    
    if verbose_2 :
        cuda.synchronize()
        current_time = getTime(current_time, 'Local stats estimated (Total)')

    # zeros init of num and den
    scale = params["scale"]
    native_imshape_y, native_imshape_x = cuda_ref_img.shape
    output_size = (round(scale*native_imshape_y), round(scale*native_imshape_x))
    num = cuda.to_device(np.zeros(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    den = cuda.to_device(np.zeros(output_size+(3,), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    
    if verbose :
        cuda.synchronize()
        getTime(t1, '\nRef Img processed (Total)')
    

    n_images = comp_imgs.shape[0]
    for im_id in range(n_images):
        if verbose :
            cuda.synchronize()
            print("\nProcessing image {} ---------\n".format(im_id+1))
            im_time = time.perf_counter()
        
        #### Moving to GPU
        cuda_img = cuda.to_device(comp_imgs[im_id])
        if verbose_3 : 
            cuda.synchronize()
            current_time = getTime(im_time, 'Arrays moved to GPU')
        
        #### Compute Grey Images
        if bayer_mode:
            cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)
            if verbose_3 :
                cuda.synchronize()
                current_time = getTime(current_time, "- grey images estimated by {}".format(grey_method))
        else:
            cuda_im_grey = cuda_img
        
        #### Block Matching
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('Beginning block matching')
        
        pre_alignment = align_image_block_matching(cuda_im_grey, reference_pyramid, options, params['block matching'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Block Matching (Total)')
            
            
        #### ICA
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nBeginning ICA alignment')
        
        cuda_final_alignment = ICA_optical_flow(
            cuda_im_grey, cuda_ref_grey, ref_gradx, ref_grady, hessian, pre_alignment, options, params['kanade'])
        
        if debug_mode:
            debug_dict["flow"].append(cuda_final_alignment.copy_to_host())
            
        if verbose_2 : 
            cuda.synchronize()
            current_time = getTime(current_time, 'Image aligned using ICA (Total)')
            
            
        #### Robustness
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nEstimating robustness')
            
        cuda_robustness = compute_robustness(cuda_img, ref_local_means, ref_local_stds, cuda_final_alignment,
                                             options, params['robustness'])
        if accumulate_r:
            add(accumulated_r, cuda_robustness)
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Robustness estimated (Total)')

            
        #### Kernel estimation
        if verbose_2 :
            cuda.synchronize()
            current_time = time.perf_counter()
            print('\nEstimating kernels')
            
        cuda_kernels = estimate_kernels(cuda_img, options, params['merging'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Kernels estimated (Total)')
            
            
        #### Merging
        if verbose_2 : 
            current_time = time.perf_counter()
            print('\nAccumulating Image')
            
        merge(cuda_img, cuda_final_alignment, cuda_kernels, cuda_robustness, num, den,
              options, params['merging'])
        
        if verbose_2 :
            cuda.synchronize()
            current_time = getTime(current_time, 'Image accumulated (Total)')
        if verbose :
            cuda.synchronize()
            getTime(im_time, '\nImage processed (Total)')
            
        if debug_mode : 
            debug_dict['robustness'].append(cuda_robustness.copy_to_host())
    
    #### Ref kernel estimation
    if verbose_2 : 
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nEstimating kernels')
        
    cuda_kernels = estimate_kernels(cuda_ref_img, options, params['merging'])
    
    if verbose_2 : 
        cuda.synchronize()
        current_time = getTime(current_time, 'Kernels estimated (Total)')
    
    #### Merge ref
    if verbose_2 :
        cuda.synchronize()
        print('\nAccumulating ref Img')
    
    if accumulate_r:     
        merge_ref(cuda_ref_img, cuda_kernels,
                  num, den,
                  options, params["merging"], accumulated_r)
    else:
        merge_ref(cuda_ref_img, cuda_kernels,
                  num, den,
                  options, params["merging"])
    
    if verbose_2 : 
        cuda.synchronize()
        getTime(current_time, 'Ref Img accumulated (Total)')
        
    # num is outwritten into num/den
    divide(num, den)
    
    if verbose_2 :
        print('\n------------------------')
        cuda.synchronize()
        current_time = getTime(current_time, 'Image normalized (Total)')
    
    if verbose :
        print('\nTotal ellapsed time : ', time.perf_counter() - t1)
    
    if accumulate_r :
        debug_dict['accumulated robustness'] = accumulated_r
        
    return num, debug_dict


def process(burst_path, options=None, custom_params=None):
    """
    Processes the burst

    Parameters
    ----------
    burst_path : str or Path
        Path of the folder where the .dng burst is located
    options : dict
        
    params : Parameters
        See params.py for more details.

    Returns
    -------
    Array
        The processed image

    """
    if options is None:
        options = {'verbose' : 0}
    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         options['verbose'] >= 1,
                                         options['verbose'] >= 2)
    params = {}
    
    # reading image stack
    ref_raw, raw_comp, ISO, tags, CFA, xyz2cam, ref_path = load_dng_burst(burst_path)
    
    # if the algorithm had to be run on a specific sensor,
    # the precise values of alpha and beta could be used instead
    if 'mode' in custom_params and custom_params['mode'] == 'grey':
        alpha = tags['Image Tag 0xC761'].values[0][0]
        beta = tags['Image Tag 0xC761'].values[1][0]
    else:
        # Averaging RGB noise values
        ## IMPORTANT NOTE : the noise model exif already are NOT for nominal ISO 100
        ## But are already scaled for the image ISO.
        alpha = sum([x[0] for x in tags['Image Tag 0xC761'].values[::2]])/3
        beta = sum([x[0] for x in tags['Image Tag 0xC761'].values[1::2]])/3
    
    #### Packing noise model related to picture ISO
    curve_iso = round_iso(ISO) # Rounds non standart ISO to regular ISO (100, 200, 400, ...)
    std_noise_model_label = 'noise_model_std_ISO_{}'.format(curve_iso)
    diff_noise_model_label = 'noise_model_diff_ISO_{}'.format(curve_iso)
    std_noise_model_path = (NOISE_MODEL_PATH / std_noise_model_label).with_suffix('.npy')
    diff_noise_model_path = (NOISE_MODEL_PATH / diff_noise_model_label).with_suffix('.npy')
    
    std_curve = np.load(std_noise_model_path)
    diff_curve = np.load(diff_noise_model_path)
    
    ## Use this to compute noise curves on the fly
    # std_curve, diff_curve = run_fast_MC(alpha, beta)
    
    
    if verbose_2:
        currentTime = getTime(currentTime, ' -- Read raw files')



    
    #### Estimating ref image SNR
    brightness = np.mean(ref_raw)
    
    id_noise = round(1000*brightness)
    std = std_curve[id_noise]
    
    SNR = brightness/std
    if verbose_1:
        print(" ",10*"-")
        print('|ISO : {}'.format(ISO))
        print('|Image brightness : {:.2f}'.format(brightness))
        print('|expected noise std : {:.2e}'.format(std))
        print('|Estimated SNR : {:.2f}'.format(SNR))
    
    SNR_params = get_params(SNR)
    
    #### Merging params dictionnaries
    
    # checking (just in case !)
    check_params_validity(SNR_params, ref_raw.shape)
    
    if custom_params is not None :
        params = merge_params(dominant=custom_params, recessive=SNR_params)
        check_params_validity(params, ref_raw.shape)
        
    #### adding metadatas to dict 
    if not 'noise' in params['merging'].keys(): 
        params['merging']['noise'] = {}

        
    params['merging']['noise']['alpha'] = alpha
    params['merging']['noise']['beta'] = beta
    
    ## Writing exifs data into parameters
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
    if not 'exif' in params['robustness'].keys(): 
        params['robustness']['exif'] = {}
        
    params['merging']['exif']['CFA Pattern'] = CFA
    params['robustness']['exif']['CFA Pattern'] = CFA
    params['ISO'] = ISO
    
    params['robustness']['std_curve'] = std_curve
    params['robustness']['diff_curve'] = diff_curve
    
    # copying parameters values in sub-dictionaries
    if 'scale' not in params["merging"].keys() :
        params["merging"]["scale"] = params["scale"]
    if 'scale' not in params['accumulated robustness denoiser'].keys() :
        params['accumulated robustness denoiser']["scale"] = params["scale"]
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
    if 'mode' not in params['accumulated robustness denoiser'].keys():
        params['accumulated robustness denoiser']["mode"] = params['mode']
    
    # deactivating robustness accumulation if robustness is disabled
    params['accumulated robustness denoiser']['median']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['gauss']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['merge']['on'] &= params['robustness']['on']
    
    params['accumulated robustness denoiser']['on'] = \
        (params['accumulated robustness denoiser']['gauss']['on'] or
         params['accumulated robustness denoiser']['median']['on'] or
         params['accumulated robustness denoiser']['merge']['on'])
     
    # if robustness aware denoiser is in merge mode, copy in merge params
    if params['accumulated robustness denoiser']['merge']['on']:
        params['merging']['accumulated robustness denoiser'] = params['accumulated robustness denoiser']['merge']
    else:
        params['merging']['accumulated robustness denoiser'] = {'on' : False}
        
        
        
    
    
    #### Running the handheld pipeline
    handheld_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), options, params)
    
    
    #### Performing frame count aware denoising if enabled
    median_params = params['accumulated robustness denoiser']['median']
    gauss_params = params['accumulated robustness denoiser']['gauss']
    
    median = median_params['on']
    gauss = gauss_params['on']
    post_frame_count_denoise = (median or gauss)
    
    params_pp = params['post processing']
    post_processing_enabled = params_pp['on']
    
    if post_frame_count_denoise or post_processing_enabled:
        if verbose_1:
            print('Beginning post processing')
    
    if post_frame_count_denoise : 
        if verbose_2:
            print('-- Robustness aware bluring')
        
        if median:
            handheld_output = frame_count_denoising_median(handheld_output, debug_dict['accumulated robustness'],
                                                           median_params)
        if gauss:
            handheld_output = frame_count_denoising_gauss(handheld_output, debug_dict['accumulated robustness'],
                                                          gauss_params)


    #### post processing
    
    if post_processing_enabled:
        if verbose_2:
            print('-- Post processing image')
        
        raw = rawpy.imread(ref_path)
        output_image = raw2rgb.postprocess(raw, handheld_output.copy_to_host(),
                                           params_pp['do color correction'],
                                           params_pp['do tonemapping'],
                                           params_pp['do gamma'],
                                           params_pp['do sharpening'],
                                           params_pp['do devignette'],
                                           xyz2cam,
                                           params_pp['sharpening']
                                           ) 
    else:
        output_image = handheld_output.copy_to_host()
        
    # Applying image orientation
    if 'Image Orientation' in tags.keys():
        ori = tags['Image Orientation'].values[0]
    else:
        ori = 1
        warnings.warns('The Image Orientation EXIF tag could not be found. \
                      The image may be mirrored or misoriented.')
    output_image = apply_orientation(output_image, ori)
    if 'accumulated robustness' in debug_dict['accumulated robustness']:
        debug_dict['accumulated robustness'] = apply_orientation(debug_dict['accumulated robustness'], ori)
    
    
    
    if params['debug']:
        return output_image, debug_dict
    else:
        return output_image
    
    
