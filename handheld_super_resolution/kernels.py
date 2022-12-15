# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

@author: jamyl
"""

from math import sqrt
import time

import numpy as np
from numba import cuda

from .linalg import get_eighen_elmts_2x2
from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, getTime


def estimate_kernels(img, options, params):
    """ Returns the kernels covariance matrix for the "img" frame, sampled at the
    center of every bayer quad (or at the center of every grey pixel in grey
    mode).

    Parameters
    ----------
    img : numpy Array[imshape_y, imshape_x]
        raw image
    options : dict
        options
    params : dict
        ['mode'] : {"bayer", "grey"}
            Wether the burst is raw or grey
        params['tuning'] : dict
            parameters driving the kernel shape

    Returns
    -------
    covs : device Array[imshape_y/2, imshape_x/2]
        covarince matrices samples at the center of each bayer quad).

    """
    t1 = time.perf_counter()
    imshape_y, imshape_x = img.shape
    
    bayer_mode = params['mode']=='bayer'
    VERBOSE = options['verbose']
    
    k_detail = params['tuning']['k_detail']
    k_denoise = params['tuning']['k_denoise']
    D_th = params['tuning']['D_th']
    D_tr = params['tuning']['D_tr']
    k_stretch = params['tuning']['k_stretch']
    k_shrink = params['tuning']['k_shrink']
    
    
    # TODO this section can be sped up with torch convolution    
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

# TODO recode this, thread separation is not good
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
        rho = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

        get_eighen_elmts_2x2(structure_tensor, l, e1, e2)

        compute_rho(l[0], l[1], rho, k_detail, k_denoise, D_th, D_tr, k_stretch,
        k_shrink)

        rho_1_sq = rho[0]*rho[0]
        rho_2_sq = rho[1]*rho[1]
        if tz == 0:
            covs[pixel_idy, pixel_idx, 0, 0] = rho_1_sq*e1[0]*e1[0] + rho_2_sq*e2[0]*e2[0]
        elif tz == 1:
            covs[pixel_idy, pixel_idx, 0, 1] = rho_1_sq*e1[0]*e1[1] + rho_2_sq*e2[0]*e2[1] 
        elif tz == 2:
            covs[pixel_idy, pixel_idx, 1, 0] = rho_1_sq*e1[0]*e1[1] + rho_2_sq*e2[0]*e2[1]
        else:
            covs[pixel_idy, pixel_idx, 1, 1] = rho_1_sq*e1[1]*e1[1] + rho_2_sq*e2[1]*e2[1]

    
@cuda.jit(device=True)
def compute_rho(l1, l2, rho, k_detail, k_denoise, D_th, D_tr, k_stretch,
                          k_shrink):
    """
    Computes rho_1 and rho_2 based on lambda1, lambda2 and the constants.

    Parameters
    ----------
    l1 : float
        lambda1 (dominant eighen value)
    l2 : float
        lambda2
    rho : shared Array[2]
        empty vector where rho_1 and rho_2 will be stored
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
    A = 1+sqrt((l1 - l2)/(l1 + l2))

    D = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1)
    

    # This is a very agressive way of driving anisotropy, but it works well so far.
    if A > 1.9:
        nu_1 = 1/k_shrink
        nu_2 = k_stretch
    else:
        nu_1 = 1
        nu_2 = 1
    # nu_1 = (k_stretch*(A -1) + (2 - A))
    # nu_2 = 1/(k_shrink*(A - 1) + (2 - A))
    
    rho[0] = k_detail*((1-D)*nu_1 + D*k_detail*k_denoise)
    rho[1] = k_detail*((1-D)*nu_2 + D*k_detail*k_denoise)


