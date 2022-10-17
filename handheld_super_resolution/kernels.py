# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

@author: jamyl
"""

from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from numba import uint8, uint16, float32, float64, cuda

from .linalg import get_eighen_elmts_2x2, invert_2x2, interpolate_cov
from .utils import clamp, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE

@cuda.jit(device=True)
def compute_harris(image, top_left_x, top_left_y, harris):
    """
    

    Parameters
    ----------
    image : shared Array[imsize_y, imsize_x]
        ref image
    top_left_x : int
        coordinate of the top left bayer pixel contained in the grey
        pixel where harris is computed
    top_left_y : int
        coordinate of the top left bayer pixel contained in the grey
        pixel where harris is computed
    harris : shared array[2, 2, 2, 2]
        zero inited array. harris[0,1] is the harris matrix in the top right 
        of the 5x5 grey patch

    """
    imshape_y, imshape_x = image.shape
    greys = cuda.shared.array((4,4), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    grads_x = cuda.shared.array((4, 3), dtype=DEFAULT_CUDA_FLOAT_TYPE)
    grads_y = cuda.shared.array((3, 4), dtype=DEFAULT_CUDA_FLOAT_TYPE)
    txp, typ, tzp = dispatch_threads_cov()
    
    #out of bounds condition
    out_of_bounds = (top_left_x-2 < 0 or top_left_x + 3 >= imshape_x or
        top_left_y-2 < 0 or top_left_y + 3 >= imshape_y)
    
    
    if (not out_of_bounds) and tzp < 2: # the 9th thread is not computing greys
        # top left bayer pixel of each grey pixel to be calculated
        top_left_grey_x = top_left_x - 2*(1-txp) + 2*tzp
        for i in range(2):
            top_left_grey_y = top_left_y - 2*(1-typ) +2*i
            
            greys[2*typ+i, 2*txp+tzp] = (image[top_left_grey_y, top_left_grey_x] +
                                         image[top_left_grey_y + 1, top_left_grey_x] +
                                         image[top_left_grey_y, top_left_grey_x + 1] +
                                         image[top_left_grey_y + 1, top_left_grey_x + 1])/4
                    
    
    cuda.syncthreads()
    if (not out_of_bounds) and tzp < 2: # the 9th thread is not computing grads
        # estimating gradients
        for i in range(2):
            grey_idx = 2*txp + tzp
            grey_idy = 2*typ + i
            if grey_idx < 3:
                grads_x[grey_idy, grey_idx] = greys[grey_idy, grey_idx + 1] - greys[grey_idy, grey_idx]
            if grey_idy < 3:
                grads_y[grey_idy, grey_idx] = greys[grey_idy + 1, grey_idx] - greys[grey_idy, grey_idx]
            
    cuda.syncthreads()
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if (not out_of_bounds):
        # averaging for estimating grad on middle point
        gradx = (grads_x[ty + 1, tx] + grads_x[ty, tx])/2
        grady = (grads_y[ty, tx +1 ] + grads_y[ty, tx])/2
        
        if  tx != 1 and ty != 1: #grad contributes to a single cov
            idy = ty//2 # 0 or 1 : harris index
            idx = tx//2
            cuda.atomic.add(harris, (idy, idx, 0, 0), gradx*gradx)
            cuda.atomic.add(harris, (idy, idx, 0, 1), gradx*grady)
            cuda.atomic.add(harris, (idy, idx, 1, 0), gradx*grady)
            cuda.atomic.add(harris, (idy, idx, 1, 1), grady*grady)
            
        if tx == 1 and ty != 1: #grad contributes to 2 covs
            for j in range(2):
                idy = ty//2
                cuda.atomic.add(harris, (idy, j, 0, 0), gradx*gradx)
                cuda.atomic.add(harris, (idy, j, 0, 1), gradx*grady)
                cuda.atomic.add(harris, (idy, j, 1, 0), gradx*grady)
                cuda.atomic.add(harris, (idy, j, 1, 1), grady*grady)
                
        if tx != 1 and ty == 1: #grad contributes to 2 covs
            for i in range(2):
                idx = tx//2
                cuda.atomic.add(harris, (i, idx, 0, 0), gradx*gradx)
                cuda.atomic.add(harris, (i, idx, 0, 1), gradx*grady)
                cuda.atomic.add(harris, (i, idx, 1, 0), gradx*grady)
                cuda.atomic.add(harris, (i, idx, 1, 1), grady*grady)
                
        else:#contributes to the 4 covs
            for i in range(2):
                for j in range(2):
                    cuda.atomic.add(harris, (i, j, 0, 0), gradx*gradx)
                    cuda.atomic.add(harris, (i, j, 0, 1), gradx*grady)
                    cuda.atomic.add(harris, (i, j, 1, 0), gradx*grady)
                    cuda.atomic.add(harris, (i, j, 1, 1), grady*grady)
        
            
            
        
            
        
@cuda.jit(device=True)
def compute_k(l1, l2, k, k_detail, k_denoise, D_th, D_tr, k_stretch,
                          k_shrink):
    """
    Computes k1 and k2 based on lambda1, lambda2 and the constants

    Parameters
    ----------
    l1 : float
        lambda1
    l2 : float
        lambda2
    k : shared Array[2]
        empty vector where k1 and k2 are stored
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
    k_stretch = 1
    k_shrink = 1
    # TODO debug
    
    A = 1+sqrt((l1 - l2)/(l1 + l2))
    D = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1)
    k_1 = k_detail*k_stretch*A
    k_2 = k_detail/(k_shrink*A)
    

    k_1 = ((1-D)*k_1 + D*k_detail*k_denoise)**2
    k_2 = ((1-D)*k_2 + D*k_detail*k_denoise)**2
    
    k[0] = k_1
    k[1] = k_2 

@cuda.jit(device=True)
def dispatch_threads_cov():
    """
    maps the 3x3 thread pool to a 2x2x2 +1 thread pool
    indexs are returned in the set {0,1}^3, with the exception of the 9th thread
    indexed [0,0,2].

    Returns
    -------
    x : int
        thread idx
    y : int
        thread idy
    z : int
        thread idz

    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if tx < 2 and ty < 2:
        return tx, ty, 0
    elif ty < 2 and tx == 2:
        return 1,ty, 1
    elif ty == 2 and tx == 0:
        return 0, 0, 1
    elif ty ==2 and tx == 1 :
        return 0, 1, 1
    else:
        return 0, 0, 2 # 9th case (tx==2, ty==2)
    
    
    
    

@cuda.jit(device=True)
def compute_kernel_covs(image, center_pos_x, center_pos_y, covs,
                       k_detail, k_denoise, D_th, D_tr, k_stretch,
                       k_shrink, DEBUG_E1, DEBUG_E2, DEBUG_L):
    """
    Computes the 4 covariance matrix around the given coordinates

    Parameters
    ----------
    image : Array[imsize_y, imsize_x]
        image containing the patch
    center_pos_x : uint
        horizontal position of the center of the Bayer 3x3 patch 
    center_pos_y : uint
        vertical position of the center of the Bayer 3x3 patch 
    covs : array[2, 2, 2, 2]
        empty arrays that will containt the covariance matrixes


    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Steps to jump from the middle bayer cell to the 3 grey neigbhours
    # it is either -1 or +1
    x_step = 2*(center_pos_x%2) - 1
    y_step = 2*(center_pos_y%2) - 1
    
    
    # coordinates of the top left pixel belonging to the center Bayer cell
    downsampled_center_pos_x = center_pos_x - center_pos_x%2
    downsampled_center_pos_y = center_pos_y - center_pos_y%2
    
    
    # Switching the thread indexing from 3x3 to 2x2x2 (+1)
    # txp, typ indicate to  which cov the thread is contributiong. tzp : each cov has 2 dedicated threads
    txp, typ, tzp = dispatch_threads_cov()
    
    # these conditions ensure that (txp = 0, typ = 0) is top left cov matrix
    if x_step >= 0 :
        thread_cov_x = downsampled_center_pos_x + txp * x_step *2
    else : 
        thread_cov_x = downsampled_center_pos_x + (1-txp) * x_step*2
        
    if y_step >= 0 :    
        thread_cov_y = downsampled_center_pos_y + typ * y_step*2
    else :
        thread_cov_y = downsampled_center_pos_y + (1-typ) * y_step*2
    
    harris = cuda.shared.array((2,2,2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    harris[txp, typ, tzp, 0] = 0 # multithreaded zero init
    harris[txp, typ, tzp, 1] = 0
    # no need to syncthreads now, because harris is used much later, after further syncthreads
    
    # Harris computed with the 9 threads
    compute_harris(image, thread_cov_x, thread_cov_y, harris)
    
    cuda.syncthreads()
    # 1 thread per cov (so 4 threads in total)
    if tzp == 0:
        l = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e1 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e2 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        k = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

        get_eighen_elmts_2x2(harris[typ, txp], l, e1, e2)

        compute_k(l[0], l[1], k, k_detail, k_denoise, D_th, D_tr, k_stretch,
        k_shrink)
    
        if tx == 0 and ty == 0: # debug
            DEBUG_E1[0] = e1[0]; DEBUG_E1[1] = e1[1]
            DEBUG_E2[0] = e2[0]; DEBUG_E2[1] = e2[1]
            DEBUG_L[0] = l[0]; DEBUG_L[1] = l[1]
        
        
        # k's are inverted compared to the original article
        k1 = k[1]
        k2 = k[0]
        covs[typ, txp, 0, 0] = k1*e1[0]*e1[0] + k2*e2[0]*e2[0]
        covs[typ, txp, 0, 1] = k1*e1[0]*e1[1] + k2*e2[0]*e2[1]
        covs[typ, txp, 1, 0] = covs[typ, txp, 0, 1]
        covs[typ, txp, 1, 1] = k1*e1[1]*e1[1] + k2*e2[1]*e2[1]
        

@cuda.jit(device=True) 
def compute_interpolated_kernel_cov(image, fine_center_pos, cov_i,
                                    k_detail, k_denoise, D_th, D_tr, k_stretch,
                                    k_shrink, DEBUG_E1, DEBUG_E2, DEBUG_L):
    """
    Computes the invert of the inteprolated covariance matrix, at a
    sub-pixel position

    Parameters
    ----------
    image : shared Array[imsize_y, imsize_x]
        image containing the patch
    fine_center_pos : shared array[2]
        y, x : subpixel position where the covariance matrix must be interpolated
    cov_i : shared array[2, 2]
        empty array that will contain the invert of the covariance matrix at
        the given position


    Returns
    -------
    None.

    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    center_pos_x = uint16(round(fine_center_pos[1]))
    center_pos_y = uint16(round(fine_center_pos[0]))
    
    # 1 Compute The 4 cov in parallel
    covs = cuda.shared.array((2, 2, 2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
    compute_kernel_covs(image, center_pos_x, center_pos_y, covs,
                        k_detail, k_denoise, D_th, D_tr, k_stretch,
                        k_shrink, DEBUG_E1, DEBUG_E2, DEBUG_L)
    cuda.syncthreads()
    
    # 2 Bilinear interpolation (single threaded)
    interpolated_cov = cuda.shared.array((2, 2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    
    if tx == 0 and ty == 0:
        interpolate_cov(covs, fine_center_pos, interpolated_cov)
    
    # 3 Inverting cov(single threaded)
    # TODO debug
        if interpolated_cov[0, 0]*interpolated_cov[1, 1] - interpolated_cov[0, 1]*interpolated_cov[1, 0] > 1e-6:
            invert_2x2(interpolated_cov, cov_i)
        # cov_i[0, 0] = interpolated_cov[0, 0]
        # cov_i[0, 1] = interpolated_cov[0, 1]
        # cov_i[1, 0] = interpolated_cov[1, 0]
        # cov_i[1, 1] = interpolated_cov[1, 1]
        # cov_i[0, 0] = 2
        # cov_i[0, 1] = 0
        # cov_i[1, 0] = 0
        # cov_i[1, 1] = 2
        
        
        else:
            # For constant luminance patch, this a dummy filter
            # TODO mayber a better patch in this case ?
            cov_i[0, 0] = 1
            cov_i[0, 1] = 0
            cov_i[1, 0] = 1
            cov_i[1, 1] = 0
    
