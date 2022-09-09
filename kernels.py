# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

@author: jamyl
"""


import numpy as np

from numba import uint8, uint16, float32, float64, cuda
from math import sqrt
from linalg import get_eighen_elmts_2x2, invert_2x2, clamp

DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

k_detail = 0.3  # [0.25, ..., 0.33]
k_denoise = 4   # [3.0, ...,5.0]
D_th = 0.05     # [0.001, ..., 0.010]
D_tr = 0.014    # [0.006, ..., 0.020]
k_stretch = 4
k_shrink = 2


@cuda.jit(device=True)
def compute_harris(image, downsampled_center_pos_x, downsampled_center_pos_y, harris, DEBUG_GREY):
    imshape_y, imshape_x = image.shape
    
    greys = cuda.shared.array((3,3), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    tx = cuda.threadIdx.x-1
    ty = cuda.threadIdx.y-1
    
    #out of bounds condition
    if (downsampled_center_pos_x-2 < 0 or downsampled_center_pos_x + 3 >= imshape_x or
        downsampled_center_pos_y-2 < 0 or downsampled_center_pos_y + 3 >= imshape_y):
        pass
    
    else:
        # top left bayer pixel
        thread_pixel_x = downsampled_center_pos_x + 2*tx
        thread_pixel_y = downsampled_center_pos_y + 2*ty
        
        greys[1+ty, 1+tx] = (image[thread_pixel_y, thread_pixel_x]/3 +
                      image[thread_pixel_y + 1, thread_pixel_x + 1]/3 +
                      image[thread_pixel_y + 1, thread_pixel_x]/6 +
                      image[thread_pixel_y, thread_pixel_x + 1]/6)
        
        #waiting the calculation of grey patch
        cuda.syncthreads()
        if tx ==0 and ty ==0:
            DEBUG_GREY[0] = greys[0, 0] 
        # estimating gradients
        grads_x = cuda.shared.array((3, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        grads_y = cuda.shared.array((2, 3 ), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        if tx < 1:
            grads_x[ty+1, tx+1] = greys[ty + 1, tx + 2] - greys[ty + 1, tx + 1]
        if ty < 1:
            grads_y[ty+1, tx+1] = greys[ty + 2, tx + 1] - greys[ty + 1, tx + 1]
            
        # waiting that all gradients are known
        cuda.syncthreads()
        if tx >=0 and ty >= 0:
            # averaging for estimating grad on middle point
            gradx = (grads_x[ty + 1, tx] + grads_x[ty, tx])/2
            grady = (grads_y[ty, tx +1 ] + grads_y[ty, tx])/2
            cuda.atomic.add(harris, (0,0), gradx*gradx)
            cuda.atomic.add(harris, (1,1), grady*grady)
            cuda.atomic.add(harris, (1,0), gradx*grady)
            cuda.atomic.add(harris, (0,1), gradx*grady)
            
            
        
@cuda.jit(device=True)
def compute_k(l1, l2, k):
    A = 1+sqrt((l1 - l2)/(l1 + l2))
    D = clamp(1 - sqrt(l1)/D_tr+D_th, 0, 1)
    k_1 = k_detail*k_stretch*A
    k_2 = k_detail/(k_shrink*A)
    
    _ = ((1-D)*k_1 + D*k_detail*k_denoise)
    k_1 = _*_
    
    _ = ((1-D)*k_2 + D*k_detail*k_denoise)
    k_2 = _*_
    
    k[0] = k_1
    k[1] = k_2 


@cuda.jit(device=True)
def compute_kernel_cov(image, center_pos_x, center_pos_y, cov_i, DEBUG_E1, DEBUG_E2, DEBUG_L, DEBUG_GRAD, DEBUG_GREY):
    """
    Returns the inverted covariance of the kernel centered at the given position

    Parameters
    ----------
    image : Array[imsize_y, imsize_x]
        image containing the patch
    center_pos_x : uint
        horizontal position of the center of the Bayer 3x3 patch 
    center_pos_y : uint
        vertical position of the center of the Bayer 3x3 patch 
    cov_i : array[2, 2]
        Inverted covariance matrix of the patch to be returned

    Returns
    -------
    None

    """
    tx = cuda.threadIdx.x-1
    ty = cuda.threadIdx.y-1
    
    # coordinates of the top left pixel belonging to the same Bayer cell
    downsampled_center_pos_x = center_pos_x - center_pos_x%2
    downsampled_center_pos_y = center_pos_y - center_pos_y%2
    
    harris = cuda.shared.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    harris[0, 0] = 0; harris[0, 1] = 0; harris[1, 0] = 0; harris[1, 1] = 0
    compute_harris(image, downsampled_center_pos_x, downsampled_center_pos_y, harris, DEBUG_GREY)
    
    l = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    e1 = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    e2 = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    k = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    cov = cuda.shared.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
    
    if tx == 0 and ty ==0 :
        get_eighen_elmts_2x2(harris, l, e1, e2)
        if l[0] + l[1] != 0:
            compute_k(l[0], l[1], k)

        DEBUG_E1[0] = e1[0]; DEBUG_E1[1] = e1[1]
        DEBUG_E2[0] = e2[0]; DEBUG_E2[1] = e2[1]
        DEBUG_L[0] = l[0]; DEBUG_L[1] = l[1]

        
    # we need k_1 and k_2 for what's next
    cuda.syncthreads()
    if l[0] + l[1] != 0:
        if tx>=0 and ty>=0:
            cov[ty, tx] = k[0]*e1[tx]*e1[ty] + k[1]*e2[tx]*e2[ty]
            
        cuda.syncthreads()
        if tx == 0 and ty ==0 :
            invert_2x2(cov, cov_i)
    else:
        # For constant luminance patch, this a dummy filter
        cov_i[0, 0] = 1
        cov_i[0, 1] = 0
        cov_i[1, 0] = 1
        cov_i[1, 1] = 0
        

        
        
        
        