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
from typing import Union, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import rawpy

from .utils_image import compute_grey_images, apply_orientation, rggb_to_cfa, estimate_image_snr
from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, divide, add, round_iso, timer
from .alignment import align, init_alignment
from .config import Config, ExifConfig
from .params import runtime_config, sanitize_config, update_snr_config
from .robustness import init_robustness, compute_robustness
from .utils_dng import load_dng_burst
from .fast_monte_carlo import run_fast_MC
from .kernels import estimate_kernels
from .merge import merge
from . import raw2rgb

NOISE_MODEL_PATH = Path(os.path.dirname(__file__)).parent / 'data' 
        

def main(ref_img: NDArray[np.float32], comp_imgs: NDArray[np.float32], config: Config) -> Tuple[NDArray, Dict[str, list]]:
    """
    This is the implementation of Alg. 1: HandheldBurstSuperResolution.
    Some part of Alg. 2: Registration are also integrated for optimisation.

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Reference frame J_1
    comp_imgs : Array[N-1, imshape_y, imshape_x]
        Remaining frames of the burst J_2, ..., J_N
        
    config : Config
        parameters.

    Returns
    -------
    num : device Array[imshape_y*s, imshape_y*s, 3]
        generated RGB image WITHOUT any post-processing.
    debug_dict : dict
        Contains (if debugging is enabled) some debugging infos.

    """
    
    grey_method = config.grey_method
    
    ### verbose and timing related stuff
    verbose = config.verbose >= 1
    verbose_2 = config.verbose >= 2
    verbose_3 = config.verbose >= 3

    init_robustness_ = timer(init_robustness, verbose_2, "\nEstimating ref image local stats", 'Local stats estimated (Total)')
    compute_grey_images_ = timer(compute_grey_images, verbose_3, end_s="- Grey images estimated by {}".format(grey_method))
    compute_robustness_ = timer(compute_robustness, verbose_2, '\nEstimating robustness', 'Robustness estimated (Total)')
    estimate_kernels_ = timer(estimate_kernels, verbose_2, '\nEstimating kernels', 'Kernels estimated (Total)')
    merge_ = timer(merge, verbose_2, '\nAccumulating Image', 'Image accumulated (Total)')
    divide_ = timer(divide, verbose_2, end_s='\n------------------------\nImage normalized (Total)')
    init_alignment_ = timer(init_alignment, verbose_2, '\nInitializing alignment', 'Alignment initialized (Total)')
    align_ = timer(align, verbose_2, '\nBeginning alignment', 'Image aligned (Total)')

    bayer_mode = config.mode=='bayer'
    debug_mode = config.debug
    debug_dict = {"robustness":[],
                  "flow":[]}

    assert config.exif is not None, "Exif data missing from conf"
    #### Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    white_balance = cuda.to_device(np.array(config.exif.white_balance))

    # This running buffer is for the image being processed
    stream = cuda.stream()
    cuda_img = cuda.device_array_like(comp_imgs[0], stream=stream)
    cuda.synchronize()
    cuda_std_curve = cuda.to_device(np.array(config.noise_model.std_curve))
    cuda_diff_curve = cuda.to_device(np.array(config.noise_model.diff_curve))
    
    if verbose :
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()

    #### Raw to grey
    if bayer_mode :
        cuda_ref_grey = compute_grey_images_(cuda_ref_img, grey_method)
    else:
        cuda_ref_grey = cuda_ref_img

    ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian = init_alignment_(cuda_ref_grey, config)

    #### Local stats estimation
    if config.robustness.enabled:
        ref_local_means, ref_local_stds = init_robustness_(cuda_ref_img, white_balance, config)
    else:
        ref_local_means, ref_local_stds = None, None

    
    accumulated_r = None
    if config.robustness.save_mask and config.robustness.enabled:
        assert ref_local_means
        accumulated_r = cuda.to_device(np.zeros(ref_local_means.shape[1:]))

    scale = config.scale
    native_imshape_y, native_imshape_x = cuda_ref_img.shape
    output_size = (round(scale*native_imshape_y), round(scale*native_imshape_x))
    # zeros init of num and den
    num = cuda.to_device(np.zeros((*output_size, 3), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    den = cuda.to_device(np.zeros((*output_size, 3), dtype = DEFAULT_NUMPY_FLOAT_TYPE))

    #### Ref kernel estimation
    cuda_kernels = estimate_kernels_(cuda_ref_img, config)
    
    ##### Merge ref
    dummy_alignment = cuda.to_device(np.zeros(ref_hessian[-1].shape[:-1], dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    dummy_r = cuda.to_device(np.ones(cuda_ref_img.shape, dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    merge_(cuda_ref_img, dummy_alignment, cuda_kernels, dummy_r, num, den, config)
    
    if verbose :
        cuda.synchronize()
        getTime(t1, '\nRef Img processed (Total)')


    for im_id in range(comp_imgs.shape[0]):
        if verbose :
            cuda.synchronize()
            print("\nProcessing image {} ---------\n".format(im_id+1))
            im_time = time.perf_counter()
        
        #### Moving to GPU
        cuda.to_device(comp_imgs[im_id], to=cuda_img, stream=stream)
        
        #### Compute Grey Images
        if bayer_mode:
            cuda_im_grey = compute_grey_images(comp_imgs[im_id], grey_method)
        else:
            cuda_im_grey = cuda_img

        final_alignment = align_(ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian,
                        cuda_im_grey, config)
        
        if debug_mode:
            debug_dict["flow"].append(final_alignment.copy_to_host())
            
        #### Robustness
        if config.robustness.enabled:
            assert ref_local_means is not None
            assert ref_local_stds is not None
            robustness = compute_robustness_(cuda_img, ref_local_means, ref_local_stds, final_alignment, white_balance, (cuda_std_curve, cuda_diff_curve), config)
        else:
            temp = np.ones_like(cuda_img, DEFAULT_NUMPY_FLOAT_TYPE)
            robustness = cuda.to_device(temp)

        if accumulated_r is not None:
            add(accumulated_r, robustness)
        
        #### Kernel estimation
        cuda_kernels = estimate_kernels_(cuda_img, config)
        
        #### Merging
        merge_(cuda_img, final_alignment, cuda_kernels, robustness, num, den, config)
        
        if verbose :
            cuda.synchronize()
            getTime(im_time, '\nImage processed (Total)')
            
        if debug_mode and config.robustness.enabled: 
            debug_dict['robustness'].append(robustness.copy_to_host())
        stream.synchronize()


        
    # num is outwritten into num/den
    divide_(num, den)
    
    if verbose :
        s = '\nTotal ellapsed time : '
        print(s, ' ' * (50 - len(s)), ': ', round((time.perf_counter() - t1), 2), 'seconds')
    
    if config.robustness.save_mask and config.robustness.enabled and accumulated_r:
        debug_dict['accumulated robustness'] = accumulated_r.copy_to_host()
        
    return num.copy_to_host(), debug_dict


def process(burst_path: Union[Path, str], config: Config):
    """
    Processes the burst

    Parameters
    ----------
    burst_path : str or Path
        Path of the folder where the .dng burst is located
    config : Config
        parameters.

    Returns
    -------
    Array
        The processed image

    """
    config = runtime_config(config)

    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         config.verbose >= 1,
                                         config.verbose >= 2)
    
    # reading image stack
    ref_raw, raw_comp, iso, tags, CFA, xyz2cam, white_balance, ref_path = load_dng_burst(burst_path)

    if config.noise_model.alpha is not None:
        # User provided custom values.
        print("Using user-provided alpha and beta values")
        alpha = config.noise_model.alpha
        beta = config.noise_model.beta
    ## The noise model exif are already scaled for the image ISO.
    elif config.mode == 'grey':
        alpha = tags['Image Tag 0xC761'].values[0][0]
        beta = tags['Image Tag 0xC761'].values[1][0]
    elif config.mode == 'bayer':
        alpha = sum([x[0] for x in tags['Image Tag 0xC761'].values[::2]])/3
        beta = sum([x[0] for x in tags['Image Tag 0xC761'].values[1::2]])/3
    else:
        raise ValueError("Noise model parameters not found in EXIF and not provided by user.")
    config.noise_model.alpha = alpha
    config.noise_model.beta = beta
    #### Packing noise model related to picture ISO
    # curve_iso = round_iso(ISO) # Rounds non standart ISO to regular ISO (100, 200, 400, ...)
    # std_noise_model_label = 'noise_model_std_ISO_{}'.format(curve_iso)
    # diff_noise_model_label = 'noise_model_diff_ISO_{}'.format(curve_iso)
    # std_noise_model_path = (NOISE_MODEL_PATH / std_noise_model_label).with_suffix('.npy')
    # diff_noise_model_path = (NOISE_MODEL_PATH / diff_noise_model_label).with_suffix('.npy')
    
    # std_curve = np.load(std_noise_model_path)
    # diff_curve = np.load(diff_noise_model_path)
    
    # Use this to compute noise curves on the fly
    std_curve, diff_curve = run_fast_MC(alpha, beta)
    
    
    if verbose_2:   
        currentTime = getTime(currentTime, ' -- Read raw files')

    #### Estimating ref image SNR
    snr = estimate_image_snr(ref_raw, alpha, beta, white_balance)

    if verbose_1:
        print(f"Estimated snr: {snr:.2f} dB")
    
    update_snr_config(config, snr)
    
    # checking (just in case !)
    sanitize_config(config, ref_raw.shape)
    

    config.exif = ExifConfig(
        cfa_pattern=CFA.tolist(),
        iso=iso,
        white_balance=list(white_balance),
    )

    config.noise_model.std_curve = std_curve.tolist()
    config.noise_model.diff_curve = diff_curve.tolist()
    
    
    #### Running the handheld pipeline
    hr_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), config)

    #### Deflip the image
    hr_output = np.moveaxis(hr_output, -1, 0)
    hr_output = rggb_to_cfa(hr_output, CFA)
    hr_output = np.moveaxis(hr_output, 0, -1)
    if 'accumulated robustness' in debug_dict:
        debug_dict['accumulated robustness'] = rggb_to_cfa(debug_dict['accumulated robustness'], CFA)
    

    #### post processing
    post_processing_enabled = config.postprocessing.enabled
    
    if post_processing_enabled:
        if verbose_2:
            print('-- Post processing image')
        
        raw = rawpy.imread(ref_path)
        hr_output = raw2rgb.postprocess(raw, hr_output,
                                           config.postprocessing.do_color_correction,
                                           config.postprocessing.do_tonemapping,
                                           config.postprocessing.do_gamma_correction,
                                           config.postprocessing.sharpening,
                                           config.postprocessing.do_devignetting,
                                           xyz2cam,
                                           ) 
        
    # Applying image orientation
    if 'Image Orientation' in tags.keys():
        ori = tags['Image Orientation'].values[0]
    else:
        ori = 1
        warnings.warn('The Image Orientation EXIF tag could not be found. \
                      The image may be mirrored or misoriented.')
        
    output_image = apply_orientation(hr_output, ori)
    if 'accumulated robustness' in debug_dict:
        debug_dict['accumulated robustness'] = apply_orientation(debug_dict['accumulated robustness'], ori)
    
    
    
    return output_image, debug_dict
