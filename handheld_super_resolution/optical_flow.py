# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

@author: jamyl
"""

from time import time

from math import floor, ceil, modf, exp, sqrt
import numpy as np
import cv2
from numba import cuda, float32, float64, int16
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states, xoroshiro128p_uniform_float32
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft2, ifft2, fftshift, ifftshift

import colour_demosaicing
import matplotlib.pyplot as plt

from .linalg import bilinear_interpolation
from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, hann, hamming, clamp
from .linalg import solve_2x2, solve_6x6_krylov, get_eighen_val_2x2
    
def init_ICA(ref_img, options, params):
    current_time, verbose, verbose_2 = time(), options['verbose'] > 1, options['verbose'] > 2


    sigma_blur = params['tuning']['sigma blur']

    tile_size = params['tuning']['tileSize']
    
    imsize_y, imsize_x = ref_img.shape
    # TODO check if this formulas are true : it shoudl match with pre alignment shape
    n_patch_y = 2*int(ceil(imsize_y/(tile_size//2)))+1
    n_patch_x = 2*int(ceil(imsize_x/(tile_size//2)))+1
    
    # Estimating gradients with separated Prewitt kernels
    kernely = np.array([[-1],
                        [0],
                        [1]])
    # kernely2 = np.array([[1, 1, 1]])
    
    kernelx = np.array([[-1,0,1]])
    # kernelx2 = np.array([[1],
    #                      [1],
    #                      [1]])

    if sigma_blur != 0:
        # 2 times gaussian 1d is faster than gaussian 2d
        temp = gaussian_filter1d(ref_img, sigma=sigma_blur, axis=-1)
        temp = gaussian_filter1d(temp, sigma=sigma_blur, axis=-2)
        
        gradx = cv2.filter2D(temp, -1, kernelx)
        # gradx = cv2.filter2D(gradx, -1, kernelx2)
        grady = cv2.filter2D(temp, -1, kernely)
        # grady = cv2.filter2D(grady, -1, kernely2)
    else:
        gradx = cv2.filter2D(ref_img, -1, kernelx)
        # gradx = cv2.filter2D(gradx, -1, kernelx2)
        grady = cv2.filter2D(ref_img, -1, kernely)
        # grady = cv2.filter2D(grady, -1, kernely2)
        
    if verbose_2:
        current_time = getTime(
            current_time, ' -- Gradients estimated')
    

    cuda_gradx = cuda.to_device(gradx)
    cuda_grady = cuda.to_device(grady)
    if verbose_2 : 
        current_time = getTime(
            current_time, ' -- Arrays moved to GPU')
        
    hessian = cuda.device_array((n_patch_y, n_patch_x, 2, 2), DEFAULT_NUMPY_FLOAT_TYPE)
    compute_hessian[(n_patch_x, n_patch_y), (tile_size, tile_size)](
        cuda_gradx, cuda_grady, tile_size, hessian)
    
    if verbose_2:
        current_time = getTime(
            current_time, ' -- Hessian estimated')
    
    return cuda_gradx, cuda_grady, hessian
    
@cuda.jit
def compute_hessian(gradx, grady, tile_size, hessian):
    imshape = gradx.shape
    patch_idx, patch_idy = cuda.blockIdx.x, cuda.blockIdx.y
    pixel_local_idx = cuda.threadIdx.x # Position relative to the patch
    pixel_local_idy = cuda.threadIdx.y
    
    pixel_global_idx = tile_size//2 * (patch_idx - 1) + pixel_local_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    pixel_global_idy = tile_size//2 * (patch_idy - 1) + pixel_local_idy
    
    inbound = (0 <= pixel_global_idy <imshape[0] and 0 <= pixel_global_idx < imshape[1])
    
    
    if pixel_local_idx < 2 and pixel_local_idy < 2:
        hessian[patch_idy, patch_idx, pixel_local_idy, pixel_local_idx] = 0 # zero init
    cuda.syncthreads()
    
    # TODO gaussian windowing ?
    if inbound : 
        local_gradx = gradx[pixel_global_idy, pixel_global_idx]
        local_grady = grady[pixel_global_idy, pixel_global_idx]
        
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 0, 0), local_gradx*local_gradx)
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 0, 1), local_gradx*local_grady)
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 1, 0), local_gradx*local_grady)
        cuda.atomic.add(hessian, (patch_idy, patch_idx, 1, 1), local_grady*local_grady)


def ICA_optical_flow(cuda_im_grey, cuda_ref_grey, cuda_gradx, cuda_grady, hessian, pre_alignment, options, params, debug = False):
    """ Computes optical flow between the ref_img and all images of comp_imgs 
    based on the ICA method (http://www.ipol.im/pub/art/2016/153/).
    The optical flow follows a translation per patch model, such that :
    ref_img(X) ~= comp_img(X + flow(X))
    

    Parameters
    ----------
    ref_img : numpy Array[imsize_y, imsize_x]
        reference image on grey level
    comp_img : numpy Array[n_images, imsize_y, imsize_x]
        images to align on grey level
    pre_alignment : numpy Array[n_images, n_tiles_y, n_tiles_x, 2]
        optical flow for each tile of each image, outputed by bloc matching.
        pre_alignment[0] must be the horizontal flow oriented towards the right if positive.
        pre_alignment[1] must be the vertical flow oriented towards the bottom if positive.
    options : dict
        options
    params : dict
        ['tuning']['kanadeIter'] : int
            Number of iterations.
        params['tuning']['tileSize'] : int
            Size of the tiles.
        params["mode"] : {"bayer", "grey"}
            Mode of the pipeline : whether the original burst is grey or raw
        params['grey method'] : {"gauss", "decimating", "FFT", "demosaicing"}
            Method that have been used to get grey images (in bayer mode)
        ['tuning']['sigma blur'] : float
            If non zero, applies a gaussian blur before computing gradients.
            
    debug : bool, optional
        If True, this function returns a list containing the flow at each iteration.
        The default is False.

    Returns
    -------
    cuda_alignment : device_array[n_images, n_tiles_y, n_tiles_x, 2]
        alignment vector for each tile of each image following the same convention
        as "pre_alignment"

    """
    if debug : 
        debug_list = []
        
    current_time, verbose, verbose_2 = time(), options['verbose'] > 1, options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']
    grey_method = params['grey method']
    bayer_mode = params["mode"]=='bayer'
    
    imsize_y, imsize_x = cuda_im_grey.shape
    n_patch_y, n_patch_x, _ = pre_alignment.shape

        
    if verbose:
        t1 = time()
        current_time = time()
        print("Estimating Lucas-Kanade's optical flow")
        
    if  bayer_mode and grey_method in ['gauss', 'decimating']:
        pre_alignment = pre_alignment/2
        # dividing by 2 because grey image is twice smaller than bayer
        # and blockmatching in bayer mode returns alignment to bayer scale
        # Note : A=A/2 is a pointer reassignment, so pre_alignment is not outwritten
        # outside the function, the local variable is simply different.
        # On the contrary, A/=2 is reassigning the values in memory, which would
        # Overwrite pre_alignment even outside of the function scope.
        
    
    # TODO this is a bit dirty. We should ensure that pre alignemnt is
    # contiguous before calling, because this is using a lot of memory.
    # pre_alignment should be outwritten by its contiguous version, whenever possible.
    # It is probably contiguous when outputted by BM, but we are swapping x and y for convention
    # coherence, breaking the contiguity.
    alignment = np.ascontiguousarray(pre_alignment).astype(DEFAULT_NUMPY_FLOAT_TYPE)
    cuda_alignment = cuda.to_device(np.ascontiguousarray(alignment))

    if verbose_2 : 
        current_time = getTime(
            current_time, ' -- Arrays moved to GPU')
    

    for iter_index in range(n_iter):
        ICA_optical_flow_iteration(
            cuda_ref_grey, cuda_gradx, cuda_grady, cuda_im_grey, cuda_alignment, hessian,
            options, params, iter_index)
        if debug :
            debug_list.append(cuda_alignment.copy_to_host())
    
    if verbose:
        getTime(t1, 'Flows estimated (Total)')
        print('\n')
        
    if debug:
        return debug_list
    return cuda_alignment

    
def ICA_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, hessian, options, params,
                               iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img : Array [imsize_y, imsize_x]
        Ref image (grey)
    gradx : Array [imsize_y, imsize_x]
        Horizontal gradient of the ref image
    grady : Array [imsize_y, imsize_x]
        Vertical gradient of the ref image
    comp_img : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference (grey images)
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The inial alignment of the tiles
    options : Dict
        Options to pass
    params : Dict
        parameters
    iter_index : int
        The iteration index (for printing evolution when verbose >2,
                             and for clearing memory)

    Returns
    -------
    new_alignment
        Array[n_images, n_tiles_y, n_tiles_x, 2]
            The adjusted tile alignment

    """
    verbose_2 = options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']
    grey_method = params['grey method']
    tile_size = params['tuning']['tileSize']

    imsize_y, imsize_x = comp_img.shape
    n_patch_y, n_patch_x, _ = alignment.shape
    
    if verbose_2 : 
        print(" -- Lucas-Kanade iteration {}".format(iter_index))
    
    current_time = time()
    double_flow = (((n_iter - 1) == iter_index) and
                   (params['mode'] == 'bayer') and
                    grey_method in ["gauss", "decimating"])
    # In bayer mode, if grey image is twice smaller than bayer (decimating mode),
    # the flow must be doubled at the last iteration to represent flow at bayer scale
    ICA_get_new_flow[[(n_patch_y, n_patch_x), (tile_size, tile_size)]
        ](ref_img, comp_img, gradsx, gradsy, alignment, hessian, tile_size,
            double_flow)   

    if verbose_2:
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')



@cuda.jit
def ICA_get_new_flow(ref_img, comp_img, gradx, grady, alignment, hessian, tile_size, upscaled_flow):
    imsize_y, imsize_x = comp_img.shape
    
    patch_idy, patch_idx = cuda.blockIdx.x, cuda.blockIdx.y
    pixel_local_idx = cuda.threadIdx.x # Position relative to the patch
    pixel_local_idy = cuda.threadIdx.y
    
    pixel_global_idx = tile_size//2 * (patch_idx - 1) + pixel_local_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    pixel_global_idy = tile_size//2 * (patch_idy - 1) + pixel_local_idy
    
    ATA = cuda.shared.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    ATB = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    
    # parallel init
    if cuda.threadIdx.x <= 1 and cuda.threadIdx.y <=1 : # copy hessian to threads
        ATA[cuda.threadIdx.y, cuda.threadIdx.x] = hessian[patch_idy, patch_idx,
                                                          cuda.threadIdx.y, cuda.threadIdx.x]
    if cuda.threadIdx.y == 2 and cuda.threadIdx.x <= 1 :
            ATB[cuda.threadIdx.x] = 0
  
    
    
    inbound = (0 <= pixel_global_idx < imsize_x) and (0 <= pixel_global_idy < imsize_y)

    if inbound :
        # Warp I with W(x; p) to compute I(W(x; p))
        new_idx = alignment[patch_idy, patch_idx, 0] + pixel_global_idx 

        
        new_idy = alignment[patch_idy, patch_idx, 1] + pixel_global_idy 
 
    
    inbound = inbound and (0 <= new_idx < imsize_x -1) and (0 <= new_idy < imsize_y - 1) # -1 for bicubic interpolation
    
    if inbound:
        # bicubic interpolation
        buffer_val = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
        pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x
        normalised_pos_x, floor_x = modf(new_idx) # https://www.rollpie.com/post/252
        normalised_pos_y, floor_y = modf(new_idy) # separating floor and floating part
        floor_x = int(floor_x)
        floor_y = int(floor_y)
        
        ceil_x = floor_x + 1
        ceil_y = floor_y + 1
        pos[0] = normalised_pos_y
        pos[1] = normalised_pos_x
        
        buffer_val[0, 0] = comp_img[floor_y, floor_x]
        buffer_val[0, 1] = comp_img[floor_y, ceil_x]
        buffer_val[1, 0] = comp_img[ceil_y, floor_x]
        buffer_val[1, 1] = comp_img[ceil_y, ceil_x]
        comp_val = bilinear_interpolation(buffer_val, pos)
        gradt = comp_val- ref_img[pixel_global_idy, pixel_global_idx]
               
        # exponentially decreasing window, of size tile_size_lk
        r_square = ((pixel_local_idx - tile_size/2)**2 +
                    (pixel_local_idy - tile_size/2)**2)
        sigma_square = (tile_size/2)**2
        w = exp(-r_square/(3*sigma_square)) # TODO this is not improving LK so much... 3 seems to be a good coef though.
        
        local_gradx = gradx[pixel_global_idy, pixel_global_idx]
        local_grady = grady[pixel_global_idy, pixel_global_idx]
        
        cuda.atomic.add(ATB, 0, -local_gradx*gradt*w)
        cuda.atomic.add(ATB, 1, -local_grady*gradt*w)

    alignment_step = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    cuda.syncthreads()
        
    if cuda.threadIdx.x <= 1 and cuda.threadIdx.y == 0:
        if abs(ATA[0, 0]*ATA[1, 1] - ATA[0, 1]*ATA[1, 0])> 1e-5: # system is solvable 
            # No racing condition, one thread for one id
            solve_2x2(ATA, ATB, alignment_step)
            alignment[patch_idy, patch_idx, cuda.threadIdx.x] += alignment_step[cuda.threadIdx.x]

            
    cuda.syncthreads()
    if pixel_local_idx == 0 and pixel_local_idy == 0: # single threaded section
        if upscaled_flow : #uspcaling because grey img is 2x smaller
            alignment[patch_idy, patch_idx, 0] *= 2
            alignment[patch_idy, patch_idx, 1] *= 2


####### Generic LK
# TODO maybe removing it
def lucas_kanade_optical_flow(ref_img, comp_img, pre_alignment, options, params, debug = False):
    """
    Computes the displacement based on a naive implementation of
    Lucas-Kanade optical flow (https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf)
    (The first method).
    The optical flow follows a translation per patch model, such that :
    ref_img(X) ~= comp_img(X + flow(X))
    

    Parameters
    ----------
    ref_img : numpy Array[imsize_y, imsize_x]
        reference image on grey level
    comp_img : numpy Array[n_images, imsize_y, imsize_x]
        images to align on grey level
    pre_alignment : numpy Array[n_images, n_tiles_y, n_tiles_x, 2]
        optical flow for each tile of each image, outputed by bloc matching.
        pre_alignment[0] must be the horizontal flow oriented towards the right if positive.
        pre_alignment[1] must be the vertical flow oriented towards the bottom if positive.
    options : dict
        options
    params : dict
        ['tuning']['kanadeIter'] : int
            Number of iterations.
        params['tuning']['tileSize'] : int
            Size of the tiles.
        params["mode"] : {"bayer", "grey"}
            Mode of the pipeline : whether the original burst is grey or raw
        params['grey method'] : {"gauss", "decimating", "FFT", "demosaicing"}
            Method that have been used to get grey images (in bayer mode)
        ['tuning']['sigma blur'] : float
            If non zero, applies a gaussian blur before computing gradients.
            
    debug : bool, optional
        If True, this function returns a list containing the flow at each iteration.
        The default is False.

    Returns
    -------
    cuda_alignment : device_array[n_images, n_tiles_y, n_tiles_x, 2]
        alignment vector for each tile of each image following the same convention
        as "pre_alignment"

    """
    if debug : 
        debug_list = []
        
    current_time, verbose, verbose_2 = time(), options['verbose'] > 1, options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']
    grey_method = params['grey method']
    sigma_blur = params['tuning']['sigma blur']
    bayer_mode = params["mode"]=='bayer'
    
    n_images, imsize_y, imsize_x = comp_img.shape
    _, n_patch_y, n_path_x, _ = pre_alignment.shape

        
    if verbose:
        t1 = time()
        current_time = time()
        print("Estimating Lucas-Kanade's optical flow")
        
    if grey_method in ['gauss', 'decimating'] and bayer_mode:
        pre_alignment = pre_alignment/2
        # dividing by 2 because grey image is twice smaller than bayer
        # and blockmatching in bayer mode returns alignment to bayer scale
        # Note : A=A/2 is a pointer reassignment, so pre_alignment is not outwritten
        # outside the function, the local variable is simply different.
        # On the contrary, A/=2 is reassigning the values in memory, which would
        # Overwrite pre_alignment even outside of the function scope.
        
    gradsx = np.empty_like(comp_img, DEFAULT_NUMPY_FLOAT_TYPE)
    gradsy = np.empty_like(comp_img, DEFAULT_NUMPY_FLOAT_TYPE)
    
    # Estimating gradients with separated Prewitt kernels
    kernely = np.array([[-1],
                        [0],
                        [1]])
    # kernely2 = np.array([[1, 1, 1]])
    
    kernelx = np.array([[-1,0,1]])
    # kernelx2 = np.array([[1],
    #                      [1],
    #                      [1]])

    for i in range(n_images):
        if sigma_blur != 0:
            # 2 times gaussian 1d is faster than gaussian 2d
            temp = gaussian_filter1d(comp_img[i], sigma=sigma_blur, axis=-1)
            temp = gaussian_filter1d(temp, sigma=sigma_blur, axis=-2)
            
            gradsx[i] = cv2.filter2D(temp, -1, kernelx)
            # gradsx[i] = cv2.filter2D(gradx, -1, kernelx2)
            gradsy[i] = cv2.filter2D(temp, -1, kernely)
            # gradsy[i] = cv2.filter2D(grady, -1, kernely2)
        else:
            gradsx[i] = cv2.filter2D(comp_img[i], -1, kernelx)
            # gradsx[i] = cv2.filter2D(gradx, -1, kernelx2)
            gradsy[i] = cv2.filter2D(comp_img[i], -1, kernely)
            # gradsy[i] = cv2.filter2D(grady, -1, kernely2)

    if verbose_2:
        current_time = getTime(
            current_time, ' -- Gradients estimated')
    

    alignment = np.ascontiguousarray(pre_alignment).astype(DEFAULT_NUMPY_FLOAT_TYPE)
    
    cuda_alignment = cuda.to_device(alignment)
    cuda_ref_img_grey = cuda.to_device(np.ascontiguousarray(ref_img))
    cuda_comp_img_grey = cuda.to_device(np.ascontiguousarray(comp_img))
    cuda_gradsx = cuda.to_device(gradsx)
    cuda_gradsy = cuda.to_device(gradsy)
    if verbose_2 : 
        current_time = getTime(
            current_time, ' -- Arrays moved to GPU')
    

    for iter_index in range(n_iter):
        lucas_kanade_optical_flow_iteration(
            cuda_ref_img_grey, cuda_gradsx, cuda_gradsy, cuda_comp_img_grey, cuda_alignment,
            options, params, iter_index)
        if debug :
            debug_list.append(cuda_alignment.copy_to_host())
    
    if verbose:
        getTime(t1, 'Flows estimated')
        print('\n')
    if debug:
        return debug_list
    return cuda_alignment


def lucas_kanade_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, options, params,
                                        iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img : Array [imsize_y, imsize_x]
        Ref image (grey)
    gradx : Array [imsize_y, imsize_x]
        Horizontal gradient of the ref image
    grady : Array [imsize_y, imsize_x]
        Vertical gradient of the ref image
    comp_img : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference (grey images)
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The inial alignment of the tiles
    options : Dict
        Options to pass
    params : Dict
        parameters
    iter_index : int
        The iteration index (for printing evolution when verbose >2,
                             and for clearing memory)

    Returns
    -------
    new_alignment
        Array[n_images, n_tiles_y, n_tiles_x, 2]
            The adjusted tile alignment

    """
    verbose_2 = options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']
    grey_method = params['grey method']
    tile_size = params['tuning']['tileSize']

    n_images, imsize_y, imsize_x = comp_img.shape
    _, n_patch_y, n_patch_x, _ = alignment.shape
    
    if verbose_2 : 
        print(" -- Lucas-Kanade iteration {}".format(iter_index))
    
    current_time = time()
    double_flow = (((n_iter - 1) == iter_index) and
                   (params['mode'] == 'bayer') and
                    grey_method in ["gauss", "decimating"])
    # In bayer mode, if grey image is twice smaller than bayer (decimating mode),
    # the flow must be doubled at the last iteration to represent flow at bayer scale
    get_new_flow[[(n_images, n_patch_y, n_patch_x), (tile_size, tile_size)]
        ](ref_img, comp_img, gradsx, gradsy, alignment, tile_size,
            double_flow)   

    
    if verbose_2:
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')





@cuda.jit
def get_new_flow(ref_img, comp_img, gradsx, gradsy, alignment, tile_size, upscaled_flow):
    n_images, imsize_y, imsize_x = comp_img.shape
    
    image_index, patch_idy, patch_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    pixel_local_idx = cuda.threadIdx.x # Position relative to the patch
    pixel_local_idy = cuda.threadIdx.y
    
    pixel_global_idx = tile_size//2 * (patch_idx - 1) + pixel_local_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    pixel_global_idy = tile_size//2 * (patch_idy - 1) + pixel_local_idy
    
    ATA = cuda.shared.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    ATB = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    
    # parallel init
    if cuda.threadIdx.x <= 1 and cuda.threadIdx.y <=1 :
        ATA[cuda.threadIdx.y, cuda.threadIdx.x] = 0
    if cuda.threadIdx.y == 2 and cuda.threadIdx.x <= 1 :
            ATB[cuda.threadIdx.x] = 0
  
    
    
    inbound = (0 <= pixel_global_idx < imsize_x) and (0 <= pixel_global_idy < imsize_y)

    if inbound :
        # Warp I with W(x; p) to compute I(W(x; p))
        new_idx = alignment[image_index, patch_idy, patch_idx, 0] + pixel_global_idx 

        
        new_idy = alignment[image_index, patch_idy, patch_idx, 1] + pixel_global_idy 
 
    
    inbound = inbound and (0 <= new_idx < imsize_x -1) and (0 <= new_idy < imsize_y - 1) # -1 for bicubic interpolation
    
    if inbound:
        # bicubic interpolation
        buffer_val = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
        pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x
        normalised_pos_x, floor_x = modf(new_idx) # https://www.rollpie.com/post/252
        normalised_pos_y, floor_y = modf(new_idy) # separating floor and floating part
        floor_x = int(floor_x)
        floor_y = int(floor_y)
        
        ceil_x = floor_x + 1
        ceil_y = floor_y + 1
        pos[0] = normalised_pos_y
        pos[1] = normalised_pos_x
        
        # Warp the gradient of I with W(x; p)
        buffer_val[0, 0] = gradsx[image_index, floor_y, floor_x]
        buffer_val[0, 1] = gradsx[image_index, floor_y, ceil_x]
        buffer_val[1, 0] = gradsx[image_index, ceil_y, floor_x]
        buffer_val[1, 1] = gradsx[image_index, ceil_y, ceil_x]
        gradx = bilinear_interpolation(buffer_val, pos)
        # gradx = gradsx[image_index, new_idy, new_idx]
        
        buffer_val[0, 0] = gradsy[image_index, floor_y, floor_x]
        buffer_val[0, 1] = gradsy[image_index, floor_y, ceil_x]
        buffer_val[1, 0] = gradsy[image_index, ceil_y, floor_x]
        buffer_val[1, 1] = gradsy[image_index, ceil_y, ceil_x]
        grady = bilinear_interpolation(buffer_val, pos)
        # grady = gradsy[image_index, new_idy, new_idx]
        
        buffer_val[0, 0] = comp_img[image_index, floor_y, floor_x]
        buffer_val[0, 1] = comp_img[image_index, floor_y, ceil_x]
        buffer_val[1, 0] = comp_img[image_index, ceil_y, floor_x]
        buffer_val[1, 1] = comp_img[image_index, ceil_y, ceil_x]
        comp_val = bilinear_interpolation(buffer_val, pos)
        gradt = comp_val- ref_img[pixel_global_idy, pixel_global_idx]
        # gradt = comp_img[image_index, new_idy, new_idx] - ref_img[pixel_global_idy, pixel_global_idx]
        
        # exponentially decreasing window, of size tile_size_lk
        r_square = ((pixel_local_idx - tile_size/2)**2 +
                    (pixel_local_idy - tile_size/2)**2)
        sigma_square = (tile_size/2)**2
        # w = exp(-r_square/(3*sigma_square)) # TODO this is not improving LK so much... 3 seems to be a good coef though.
        w = 1
        cuda.atomic.add(ATB, 0, -gradx*gradt*w)
        cuda.atomic.add(ATB, 1, -grady*gradt*w)

        
        # Compute the Hessian matrix
        cuda.atomic.add(ATA, (0, 0), gradx*gradx*w)
        cuda.atomic.add(ATA, (0, 1), gradx*grady*w)
        cuda.atomic.add(ATA, (1, 0), gradx*grady*w)
        cuda.atomic.add(ATA, (1, 1), grady*grady*w)
                    

    alignment_step = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    cuda.syncthreads()
        
    if cuda.threadIdx.x <= 1 and cuda.threadIdx.y == 0:
        if abs(ATA[0, 0]*ATA[1, 1] - ATA[0, 1]*ATA[1, 0])> 1e-5: # system is solvable 
            # No racing condition, one thread for one id
            solve_2x2(ATA, ATB, alignment_step)
            alignment[image_index, patch_idy, patch_idx, cuda.threadIdx.x] += alignment_step[cuda.threadIdx.x]

            
    cuda.syncthreads()
    if pixel_local_idx == 0 and pixel_local_idy == 0: # single threaded section
        if upscaled_flow : #uspcaling because grey img is 2x smaller
            alignment[image_index, patch_idy, patch_idx, 0] *= 2
            alignment[image_index, patch_idy, patch_idx, 1] *= 2









# TODO this can be rewritten but it is kept for compatibility with affinity method
# @cuda.jit(device=True, inline=True)
# def warp_flow_x(pos, local_flow):
#     return local_flow[0]

# @cuda.jit(device=True, inline=True)
# def warp_flow_y(pos, local_flow):
#     return local_flow[1]


@cuda.jit(device=True)
def get_closest_flow(idx_sub, idy_sub, optical_flows, tile_size, imsize, local_flow):
    # TODO windowing is only applied to pixels which are on 4 overlapped tile.
    # The corner and side condition remain with a generic square window 
    """
    Returns the estimated optical flow for a subpixel (or a pixel), based on
    the tile based estimated optical flow. Note that this function can be called
    either by giving id's relative to a grey scale, with tile_size being the
    number of grey pixels on the side of a tile, or by giving id's relative to a bayer
    scale, with tile_size being the number of bayer pixels on the side of a tile. Imsize simply needs 
    to be coherent with the choice.

    Parameters
    ----------
    idx_sub, idy_sub : float
        subpixel ids where the flow must be estimated
    optical_flow : Array[n_tiles_y, n_tiles_x, 6]
        tile based optical flow


    Returns
    -------
    flow : array[2]
        optical flow at the given point

    """
    imshape = optical_flows.shape[:2]
    pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    pos[0] = round(idy_sub)
    pos[1] = round(idx_sub)
    patch_idy_bottom = int((idy_sub + (tile_size//2))//(tile_size//2))
    patch_idy_top = patch_idy_bottom - 1
 
    patch_idx_right = int((idx_sub + (tile_size//2))//(tile_size//2))
    patch_idx_left = patch_idx_right - 1





    patch_idx_right = clamp(patch_idx_right, 0, imshape[1]-1)
    patch_idx_left = clamp(patch_idx_left, 0, imshape[1]-1)
    
    patch_idy_top = clamp(patch_idx_right, 0, imshape[0]-1)
    patch_idx_bottom = clamp(patch_idx_right, 0, imshape[0]-1)

    # Index out of bound. With zero flow, they will be discarded later
    if (idx_sub < 0 or idx_sub >= imsize[1] or
        idy_sub < 0 or idy_sub >= imsize[0]):
        flow_x = 0
        flow_y = 0
    
    
    
    tr = optical_flows[patch_idy_top, patch_idx_right]
    tl = optical_flows[patch_idy_top, patch_idx_left]
    br = optical_flows[patch_idy_bottom, patch_idx_right]
    bl = optical_flows[patch_idy_bottom, patch_idx_left]
    # tl = bl= br = tr
    
    flow_x = 0.25*(tr[0] + bl[0] + br[0] + tl[0])
    flow_y= 0.25*(tr[1] + bl[1] + br[1] + tl[1])
    
    local_flow[0] = flow_x
    local_flow[1] = flow_y
    
    # Side condition when the required patch is not existing (because it would be 
    # too small for block matching)
    # elif patch_idy_top >= imshape[0] and patch_idx_left >= imshape[1]:
    #     flow_x = warp_flow_x(pos, optical_flows[-1, -1])
    #     flow_y = warp_flow_y(pos, optical_flows[-1, -1])
        
    # elif patch_idy_top >= imshape[0]:
    #     if patch_idx_left >= 0 and patch_idx_right <imshape[1]:
    #         flow_x = (warp_flow_x(pos, optical_flows[-1, patch_idx_left]) + warp_flow_x(pos, optical_flows[-1, patch_idx_right]))/2
    #         flow_y = (warp_flow_y(pos, optical_flows[-1, patch_idx_left]) + warp_flow_y(pos, optical_flows[-1, patch_idx_right]))/2
    #     elif patch_idx_left >= 0 :
    #         flow_x = warp_flow_x(pos, optical_flows[-1, patch_idx_left])
    #         flow_y = warp_flow_y(pos, optical_flows[-1, patch_idx_left])
    #     elif patch_idx_right < imshape[1] :
    #         flow_x = warp_flow_x(pos, optical_flows[-1, patch_idx_right])
    #         flow_y = warp_flow_y(pos, optical_flows[-1, patch_idx_right])

    # elif patch_idx_left >= imshape[1]:
    #     if patch_idy_top >= 0 and patch_idy_bottom < imshape[0]:
    #         flow_x = (warp_flow_x(pos, optical_flows[patch_idy_top, -1]) + warp_flow_x(pos, optical_flows[patch_idy_bottom, -1]))/2
    #         flow_y = (warp_flow_y(pos, optical_flows[patch_idy_top, -1]) + warp_flow_y(pos, optical_flows[patch_idy_bottom, -1]))/2
    #     elif patch_idy_top >= 0 :
    #         flow_x = warp_flow_x(pos, optical_flows[patch_idy_top, -1])
    #         flow_y = warp_flow_y(pos, optical_flows[patch_idy_top, -1])
    #     elif patch_idy_bottom < imshape[1] :
    #         flow_x = warp_flow_x(pos, optical_flows[patch_idy_bottom, -1])
    #         flow_y = warp_flow_y(pos, optical_flows[patch_idy_bottom, -1])
            
            
    # # corner conditions
    # elif patch_idy_bottom >= imshape[0] and patch_idx_left < 0:
    #     flow_x = warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right])
    #     flow_y = warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right])

    # elif patch_idy_bottom >= imshape[0] and patch_idx_right >= imshape[1]:
    #     flow_x = warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left])
    #     flow_y = warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left])

    # elif patch_idy_top < 0 and patch_idx_left < 0:
    #     flow_x = warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right])
    #     flow_y = warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right])

    # elif patch_idy_top < 0 and patch_idx_right >= imshape[1]:
    #     flow_x = warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left])
    #     flow_y = warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left])

    # # side conditions
    # elif patch_idy_bottom >= imshape[0]:
    #     flow_x = (warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left]) +
    #               warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right]))/2
    #     flow_y = (warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left]) +
    #               warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right]))/2

    # elif patch_idy_top < 0:
    #     flow_x = (warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
    #             warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right]))/2
    #     flow_y = (warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
    #             warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right]))/2

    # elif patch_idx_left < 0:
    #     flow_x = (warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right]) +
    #             warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right]))/2
    #     flow_y = (warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right]) +
    #             warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right]))/2

    # elif patch_idx_right >= imshape[1]:
    #     flow_x = (warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
    #             warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left]))/2
    #     flow_y = (warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
    #             warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left]))/2

    # # general case
    # else:
    #     # Averaging patches with a window function
    #     tl = hamming(idy_sub%(tile_size/2), idx_sub%(tile_size/2), tile_size)
    #     tr = hamming(idy_sub%(tile_size/2), (tile_size/2) - idx_sub%(tile_size/2), tile_size)
    #     bl = hamming((tile_size/2) - idy_sub%(tile_size/2), idx_sub%(tile_size/2), tile_size)
    #     br = hamming((tile_size/2) - idy_sub%(tile_size/2), (tile_size/2) - idx_sub%(tile_size/2), tile_size)
        
    #     k = tl + tr + br + bl
        
        
    #     flow_x = (warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left])*tl +
    #               warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right])*tr +
    #               warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left])*bl +
    #               warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right])*br)/k
    #     flow_y = (warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left])*tl +
    #               warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right])*tr +
    #               warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left])*bl +
    #               warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right])*br)/k
    

    local_flow[0] = flow_x
    local_flow[1] = flow_y
