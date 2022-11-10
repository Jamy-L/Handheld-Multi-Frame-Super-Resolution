# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""


from time import time
import math

import numpy as np
from numba import uint8, uint16, int16, float32, float64, jit, njit, cuda, int32

from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, EPSILON
from .optical_flow import get_closest_flow
from .kernels import interpolate_cov
from .linalg import quad_mat_prod, invert_2x2


def merge(ref_img, comp_imgs, alignments, covs, r, options, params):
    """
    Merges all the images, based on the alignments previously estimated.
    The size of the merge_result is adjustable with params['scale']


    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    comp_imgs : Array [n_images,imsize_y, imsize_x]
        The compared images
    alignments : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment
    r : Device_Array[n_images, imsize_y/2, imsize_x/2, 3]
            Robustness of the moving images
    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    merge_result : Array[scale * imsize_y, scale * imsize_x, 3]
        merged images

    """
    VERBOSE = options['verbose']
    SCALE = params['scale']
    
    CFA_pattern = cuda.to_device(params['exif']['CFA Pattern'])
    bayer_mode = params['mode'] == 'bayer'
    act = params['kernel'] == 'act'
    if bayer_mode : 
        TILE_SIZE = params['tuning']['tileSize']*2
    else:
        TILE_SIZE = params['tuning']['tileSize']
    
    
    N_IMAGES, N_TILES_Y, N_TILES_X, _ \
        = alignments.shape

    if VERBOSE > 1:
        print('Beginning merge process')
        current_time = time()

    native_im_size = ref_img.shape
    # casting to integer to account for floating scale
    output_size = (round(SCALE*native_im_size[0]), round(SCALE*native_im_size[1]))
    output_img = cuda.device_array(output_size+(19 + 10*N_IMAGES,), dtype = DEFAULT_NUMPY_FLOAT_TYPE) #third dim for rgb channel
    # TODO 3 channels are enough, the rest is for debugging


    # specifying the block size
    # 1 block per output pixel, 9 threads per block
    threadsperblock = (3, 3)
    # we need to swap the shape to have idx horiztonal
    blockspergrid = (output_size[1], output_size[0])
                    
    current_time = time()

    accumulate[blockspergrid, threadsperblock](
        ref_img, comp_imgs, alignments, covs, r,
        bayer_mode, act, SCALE, TILE_SIZE, CFA_pattern,
        output_img)
    cuda.synchronize()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Data merged on GPU side')

    
    merge_result = output_img.copy_to_host()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Data returned from GPU')

    return merge_result

@cuda.jit(device=True)
def get_channel(patch_pixel_idx, patch_pixel_idy, CFA_pattern):
    """
    Return 0, 1 or 2 depending if the coordinates point a red, green or
    blue pixel on the Bayer frame

    Parameters
    ----------
    patch_pixel_idx : unsigned int
        horizontal coordinates
    patch_pixel_idy : unigned int
        vertical coordinates

    Returns
    -------
    int

    """
    return uint8(CFA_pattern[patch_pixel_idy%2, patch_pixel_idx%2])

@cuda.jit
def accumulate(ref_img, comp_imgs, alignments, covs, r,
               bayer_mode, act, scale, tile_size, CFA_pattern,
               output_img):
    """
    Cuda kernel, each block represents an output pixel. Each block contains
    a 3 by 3 neighborhood for each moving image. A single threads takes
    care of one of these pixels, for all the moving images.



    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    comp_imgs : Array[n_images, imsize_y, imsize_x]
        The compared images
    alignements : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The alignemtn vectors for each tile of each image
    r : Array[n_images, imsize_y/2, imsize_x/2, 3]
        Robustness of the moving images
    output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
        The empty output image

    Returns
    -------
    None.

    """

    output_pixel_idx, output_pixel_idy = cuda.blockIdx.x, cuda.blockIdx.y
    tx = cuda.threadIdx.x-1
    ty = cuda.threadIdx.y-1
    output_size_y, output_size_x, _ = output_img.shape
    n_images, input_size_y, input_size_x = comp_imgs.shape
    input_imsize = (input_size_y, input_size_x)
    
    if bayer_mode:
        acc = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        acc = cuda.shared.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.shared.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    
    coarse_ref_sub_pos = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # y, x
    
    # single Threaded section
    if tx == 0 and ty == 0:
        
        coarse_ref_sub_pos[0] = output_pixel_idy / scale          
        coarse_ref_sub_pos[1] = output_pixel_idx / scale
        
        acc[0] = 0
        if bayer_mode : 
            acc[1] = 0
            acc[2] = 0

        val[0] = 0
        if bayer_mode:
            val[1] = 0
            val[2] = 0

    patch_center_pos = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x
    local_optical_flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    for image_index in range(n_images + 1):
        if tx == 0 and ty == 0: # Single threaded fetch of the flow
            if image_index == 0:  # ref image
                # no optical flow
                local_optical_flow[0] = 0
                local_optical_flow[1] = 0

            else:
                get_closest_flow(coarse_ref_sub_pos[1], # flow is x, y and pos is y, x
                                  coarse_ref_sub_pos[0],
                                  alignments[image_index - 1],
                                  tile_size,
                                  input_imsize,
                                  local_optical_flow)
                
                
            patch_center_pos[1] = coarse_ref_sub_pos[1] + local_optical_flow[0]
            patch_center_pos[0] = coarse_ref_sub_pos[0] + local_optical_flow[1]
        
        # we need the position of the patch before computing the kernel
        cuda.syncthreads()

        interpolated_cov = cuda.shared.array((2, 2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cov_i = cuda.shared.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
    
        
        # coordinates of the top left bayer pixel in each of the 9
        # neigbhoors bayer cells
        
        thread_pixel_idx = round(patch_center_pos[1]) + tx
        thread_pixel_idy = round(patch_center_pos[0]) + ty
        
        # computing kernel
        if not act:
            
            # fetching the 4 closest covs
            close_covs = cuda.shared.array((2, 2, 2 ,2), DEFAULT_CUDA_FLOAT_TYPE)
            grey_pos = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
            if bayer_mode and tx==0 and ty==0:
                grey_pos[0] = (patch_center_pos[0]-0.5)/2 # grey grid is offseted and twice more sparse
                grey_pos[1] = (patch_center_pos[1]-0.5)/2
                
            elif tx==0 and ty==0:
                grey_pos[0] = patch_center_pos[0]
                grey_pos[1] = patch_center_pos[1]
            
            cuda.syncthreads()
            if tx >= 0 and ty >= 0: # TODO sides can get negative grey indexes. It leads to weird covs.
                close_covs[0, 0, ty, tx] = covs[image_index, 
                                                int(math.floor(grey_pos[0])),
                                                int(math.floor(grey_pos[1])),
                                                ty, tx]
                close_covs[0, 1, ty, tx] = covs[image_index,
                                                int(math.floor(grey_pos[0])),
                                                int(math.ceil(grey_pos[1])),
                                                ty, tx]
                close_covs[1, 0, ty, tx] = covs[image_index,
                                                int(math.ceil(grey_pos[0])),
                                                int(math.floor(grey_pos[1])),
                                                ty, tx]
                close_covs[1, 1, ty, tx] = covs[image_index,
                                                int(math.ceil(grey_pos[0])),
                                                int(math.ceil(grey_pos[1])),
                                                ty, tx]
            cuda.syncthreads()
            # interpolating covs at the desired spot
            if tx == 0 and ty == 0: # single threaded interpolation # TODO we may parallelize later
                interpolate_cov(close_covs, grey_pos, interpolated_cov)
                
                
                if abs(interpolated_cov[0, 0]*interpolated_cov[1, 1] - interpolated_cov[0, 1]*interpolated_cov[1, 0]) > 1e-6: # checking if cov is invertible
                    invert_2x2(interpolated_cov, cov_i)
                else:
                    cov_i[0, 0] = 1
                    cov_i[0, 1] = 0
                    cov_i[1, 0] = 0
                    cov_i[1, 1] = 1
                    
        
        cuda.syncthreads()
        
        
        
        # checking if pixel is r, g or b
        if bayer_mode : 
            channel = get_channel(thread_pixel_idx,
                                  thread_pixel_idy,
                                  CFA_pattern)
        else:
            channel = 0
        
        
        # fetching robustness
        if image_index == 0:
            local_r = 1 # for all 9 threads and each 4 pixels
        elif 0 <= thread_pixel_idx < input_size_x - 1 and 0 <= thread_pixel_idx < input_size_y - 1: # inbound
            if bayer_mode : 
                local_r = r[image_index - 1,
                            int((coarse_ref_sub_pos[0] + ty-0.5)/2),
                            int((coarse_ref_sub_pos[1] + tx-0.5)/2)]

            else:
                local_r = r[image_index - 1,
                            int(coarse_ref_sub_pos[0] + ty),
                            int(coarse_ref_sub_pos[1] + tx)]
            


        # in bounds conditions
        if 0 <= thread_pixel_idx < input_size_x and 0 <= thread_pixel_idy < input_size_y: # inbound
            if image_index == 0:
                c = ref_img[thread_pixel_idy, thread_pixel_idx]
            else:
                c = comp_imgs[image_index - 1, thread_pixel_idy, thread_pixel_idx]


        dist = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        # applying invert transformation and upscaling
        fine_sub_pos_x = scale * (thread_pixel_idx - local_optical_flow[0])
        fine_sub_pos_y = scale * (thread_pixel_idy - local_optical_flow[1])
        dist[0] = (fine_sub_pos_x - output_pixel_idx)
        dist[1] = (fine_sub_pos_y - output_pixel_idy)


    
        # TODO Debugging
        if tx==0 and ty == 0 :
            output_img[output_pixel_idy, output_pixel_idx, 3 + image_index*3 + 0] = dist[0]
            output_img[output_pixel_idy, output_pixel_idx, 3 + image_index*3 + 1] = dist[1]
            output_img[output_pixel_idy, output_pixel_idx, 3 + image_index*3 + 2] = 2*thread_pixel_idy%2 + thread_pixel_idx
    
        if act : 
            y = max(0, 2*(dist[0]*dist[0] + dist[1]*dist[1]))
        else:
            y = max(0, quad_mat_prod(cov_i, dist))
            # y can be slightly negative because of numerical precision.
            # I clamp it to not explode the error with exp
        if bayer_mode : 
            w = math.exp(-0.5*y/scale**2)
        else:
            w = math.exp(-0.5*4*y/scale**2) # original kernel constants are designed for bayer distances, not greys.
        
        cuda.atomic.add(val, channel, c*w*local_r)
        cuda.atomic.add(acc, channel, w*local_r)
            
        # We need to wait the accumulation of all the pixels before going for
        # the next image, because sharred arrays will be overwritten
        cuda.syncthreads()
    if tx == 0 and ty == 0:
        for chan in range(3): # TODO bayer case
            output_img[output_pixel_idy, output_pixel_idx, chan] = val[chan]/(acc[chan] + EPSILON) 