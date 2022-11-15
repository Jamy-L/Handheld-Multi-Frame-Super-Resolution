# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

@author: jamyl
"""

from math import sqrt
from time import time

import numpy as np
import matplotlib.pyplot as plt
from numba import uint8, uint16, float32, float64, cuda

from .linalg import get_eighen_elmts_2x2, invert_2x2, interpolate_cov
from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, getTime


@cuda.jit(device=True)
def compute_k(l1, l2, k, k_detail, k_denoise, D_th, D_tr, k_stretch,
                          k_shrink):
    """
    Computes k1 and k2 based on lambda1, lambda2 and the constants

    Parameters
    ----------
    l1 : float
        lambda1 (dominant eighen value)
    l2 : float
        lambda2
    k : shared Array[2]
        empty vector where k1 and k2 will be stored
    k_detail : TYPE
        DESCRIPTION.
    k_denoise : TYPE
        DESCRIPTION.
    D_th : TYPE
        DESCRIPTION.
    D_tr : TYPE
        DESCRIPTION.
    k_stretch : TYPE
        DESCRIPTION.
    k_shrink : TYPE
        DESCRIPTION.


    """
    # k_stretch = 1
    # k_shrink = 1
    
    k_stretch = 4
    k_shrink = 2
    
    # TODO debug
    
    A = 1+sqrt((l1 - l2)/(l1 + l2))
    # This is a very agressive way of driving anisotropy, but it works well so far.
    if A <1.9:
        A = 1
    else:
        A = 2

    D = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1)
    
    # D = 0
    k_1 = k_detail* (k_stretch*(A -1) + (2 - A))
    k_2 = k_detail/(k_shrink*(A - 1) + (2 - A))
    

    k_1 = ((1-D)*k_1 + D*k_detail*k_denoise)**2
    k_2 = ((1-D)*k_2 + D*k_detail*k_denoise)**2
    
    
    k[0] = k_1
    k[1] = k_2 






def estimate_kernels(img, options, params):
    """ Returns the kernels covariance matrix for each frame, sampled at the
    center of every bayer quad (or at the center of every grey pixel in grey
    mode).

    Parameters
    ----------
    ref_img : numpy Array[imshape_y, imshape_x]
        raw reference image
    comp_imgs : numpy Array[n_images, imshape_y, imshape_x]
        raw images of the burst
    options : dict
        options
    params : dict
        ['mode'] : {"bayer", "grey"}
            Wether the burst is raw or grey
        params['tuning'] : dict
            parameters driving the kernel shape

    Returns
    -------
    covs : device Array[n_images+1, imshape_y/2, imshape_x/2]
        covarince matrices samples at the center of each bayer quad, for each
        frame (including reference).

    """
    t1 = time()
    imshape_y, imshape_x = img.shape
    
    bayer_mode = params['mode']=='bayer'
    VERBOSE = options['verbose']
    
    k_detail = params['tuning']['k_detail']
    k_denoise = params['tuning']['k_denoise']
    D_th = params['tuning']['D_th']
    D_tr = params['tuning']['D_tr']
    k_stretch = params['tuning']['k_stretch']
    k_shrink = params['tuning']['k_shrink']
    

    
    if bayer_mode : 
        # grey level
        img_grey = (img[::2, ::2] + img[1::2, 1::2] + img[::2, 1::2] + img[1::2, ::2])/4
    
        if VERBOSE>2:
            t1 = getTime(t1, "- Decimated Image")
    else :
        img_grey = img # no need to copy now, they will be copied to gpu later.

    covs = cuda.device_array((img_grey.shape[0],
                              img_grey.shape[1], 2,2), DEFAULT_NUMPY_FLOAT_TYPE)
    
    grey_imshape_y, grey_imshape_x = img_grey.shape
    
    # TODO The method is good but the implementation seems dirty. Maybe a 
    # clean convolution with cv2 would be faster?
    gradsx = np.empty((grey_imshape_y, grey_imshape_x-1), DEFAULT_NUMPY_FLOAT_TYPE)
    gradsx = img_grey[:,1:] - img_grey[:,:-1]

    
    gradsy = np.empty((grey_imshape_y-1, grey_imshape_x), DEFAULT_NUMPY_FLOAT_TYPE)
    gradsy = img_grey[1:,:] - img_grey[:-1,:]

    
    cuda_gradsx = cuda.to_device(gradsx/2)
    cuda_gradsy = cuda.to_device(gradsy/2)

    if VERBOSE>2:
        t1 = getTime(t1, "- Gradients computed")

    cuda_estimate_kernel[(grey_imshape_x, grey_imshape_y),
                         (2, 2, 4)](cuda_gradsx, cuda_gradsy,
                                    k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink,
                                    covs)  
    
    return covs
    
@cuda.jit
def cuda_estimate_kernel(gradsx, gradsy,
                         k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink,
                         covs):
    pixel_idx, pixel_idy = cuda.blockIdx.x, cuda.blockIdx.y 
    tx = cuda.threadIdx.x # tx, ty for the 4 gradient point
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z #tz for the cov 4 coefs
    
    imshape_y, imshape_x, _, _ = covs.shape
    
    
    structure_tensor = cuda.shared.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    grad = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) #x, y
    
    # top left pixel among the 4 that accumulates for 1 grad
    thread_pixel_idx = pixel_idx - 1 + tx
    thread_pixel_idy = pixel_idy - 1 + ty
    
    inbound = (0 < thread_pixel_idy < imshape_y-1) and (0 < thread_pixel_idx < imshape_x-1)
    
    
    if tz == 0:
        structure_tensor[ty, tx] = 0 # multithreaded zero init

        
    cuda.syncthreads()
    if inbound :
        grad[0] = (gradsx[thread_pixel_idy+1, thread_pixel_idx] + gradsx[thread_pixel_idy, thread_pixel_idx])/2
        grad[1] = (gradsy[thread_pixel_idy, thread_pixel_idx+1] + gradsy[thread_pixel_idy, thread_pixel_idx])/2
    
    
    
    # each 4 cov coefs are processed in parallel, for each 4 gradient point (2 parallelisations)
    if tz == 0 and inbound:
        cuda.atomic.add(structure_tensor, (0, 0), grad[0]*grad[0])
    elif tz == 1 and inbound:
        cuda.atomic.add(structure_tensor, (0, 1), grad[0]*grad[1])
    elif tz == 2 and inbound:
        cuda.atomic.add(structure_tensor, (1, 0), grad[0]*grad[1])
    elif inbound:
        cuda.atomic.add(structure_tensor, (1, 1), grad[1]*grad[1])
    
    cuda.syncthreads()
    # structure tensor is computed at this point. Now calculating covs

    
    
    if (tx == 0 and ty ==0): # TODO maybe we can optimize this single threaded section
        l = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e1 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e2 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        k = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

        get_eighen_elmts_2x2(structure_tensor, l, e1, e2)

        compute_k(l[0], l[1], k, k_detail, k_denoise, D_th, D_tr, k_stretch,
        k_shrink)

        # k's are inverted compared to the original article
        k1 = k[1]
        k2 = k[0]
        if tz == 0:
            # covs[image_index, pixel_idy, pixel_idx, 0, 0] = structure_tensor[0, 0]
            # covs[pixel_idy, pixel_idx, 0, 0] = e1[0] # eighen vectors
            # covs[pixel_idy, pixel_idx, 0, 0] = clamp(1 - sqrt(l[0])/D_tr + D_th, 0, 1)
            covs[pixel_idy, pixel_idx, 0, 0] = k1*e1[0]*e1[0] + k2*e2[0]*e2[0]
        elif tz == 1:
            # covs[pixel_idy, pixel_idx, 0, 1] = structure_tensor[0, 1]
            # covs[pixel_idy, pixel_idx, 0, 1] = e2[0]
            # covs[pixel_idy, pixel_idx, 0, 1] = 1+sqrt((l[0] - l[1])/(l[0] + l[1]))
            covs[pixel_idy, pixel_idx, 0, 1] = k1*e1[0]*e1[1] + k2*e2[0]*e2[1]
            
        elif tz == 2:
            # covs[pixel_idy, pixel_idx, 1, 0] = structure_tensor[1, 0]
            # covs[pixel_idy, pixel_idx, 1, 0] = e1[1]
            # covs[pixel_idy, pixel_idx, 1, 0] = l[0]
            covs[pixel_idy, pixel_idx, 1, 0] = k1*e1[0]*e1[1] + k2*e2[0]*e2[1]
        else:
            # covs[pixel_idy, pixel_idx, 1, 1] = structure_tensor[1, 1]
            # covs[pixel_idy, pixel_idx, 1, 1] = e2[1]
            covs[pixel_idy, pixel_idx, 1, 1] = k1*e1[1]*e1[1] + k2*e2[1]*e2[1]