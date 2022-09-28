# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""

from optical_flow import get_closest_flow_V2
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_interpolated_kernel_cov
from linalg import quad_mat_prod
from robustness import fetch_robustness, compute_robustness

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
from time import time
import cupy as cp
from scipy.interpolate import interp2d
import math
from torch import from_numpy

from fast_two_stage_psf_correction.fast_optics_correction.raw2rgb import process_isp

EPSILON = 1e-6
DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32


def merge(ref_img, comp_imgs, alignments, r, options, params):
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
    # TODO Non int scale is broken atm
    VERBOSE = options['verbose']
    SCALE = params['scale']
    
    CFA_pattern = params['exif']['CFA Pattern']
    
    TILE_SIZE = params['tuning']['tileSizes']
    k_detail =params['tuning']['k_detail']
    k_denoise = params['tuning']['k_denoise']
    D_th = params['tuning']['D_th']
    D_tr = params['tuning']['D_tr']
    k_stretch = params['tuning']['k_stretch']
    k_shrink = params['tuning']['k_shrink']
    
    
    N_IMAGES, N_TILES_Y, N_TILES_X, _ \
        = alignments.shape

    if VERBOSE > 2:
        print('Beginning merge process')
        current_time = time()

    native_im_size = ref_img.shape
    output_size = (SCALE*native_im_size[0], SCALE*native_im_size[1])
    output_img = cuda.device_array(output_size+(31,), dtype = DEFAULT_NUMPY_FLOAT_TYPE) #third dim for rgb channel
    # TODO 3 channels are enough, the rest is for debugging
    # we may also chose uint16 for output img, but float is nice for debugging

    # specifying the block size
    # 1 block per output pixel, 9 threads per block
    threadsperblock = (3, 3)
    # we need to swap the shape to have idx horiztonal
    blockspergrid = (output_size[1], output_size[0])

    @cuda.jit(device=True)
    def get_channel(patch_pixel_idx, patch_pixel_idy):
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
    def accumulate(ref_img, comp_imgs, alignments, r, output_img):
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
        input_size_y, input_size_x = ref_img.shape
        
        acc = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.shared.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        
        coarse_ref_sub_pos = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # y, x
        
        # single Threaded section
        if tx == 0 and ty == 0:
            
            coarse_ref_sub_pos[0] = output_pixel_idy / SCALE          
            coarse_ref_sub_pos[1] = output_pixel_idx / SCALE
            
            acc[0] = 0
            acc[1] = 0
            acc[2] = 0

            val[0] = 0
            val[1] = 0
            val[2] = 0

        patch_center_pos = cuda.shared.array(2, uint16) # y, x
        local_optical_flow = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

        for image_index in range(N_IMAGES + 1):
            if tx == 0 and ty == 0: # Single threaded fetch of the flow
                if image_index == 0:  # ref image
                    # no optical flow
                    patch_center_pos[1] = coarse_ref_sub_pos[0]
                    patch_center_pos[0] = coarse_ref_sub_pos[0]

                else:
                    get_closest_flow_V2(coarse_ref_sub_pos[1], # flow is x, y and pos is y, x
                                      coarse_ref_sub_pos[0],
                                      alignments[image_index - 1],
                                      TILE_SIZE,
                                      native_im_size,
                                      local_optical_flow)
                    
                    
                    patch_center_pos[1] = coarse_ref_sub_pos[1] + local_optical_flow[0]
                    patch_center_pos[0] = coarse_ref_sub_pos[0] + local_optical_flow[1]
            
            # we need the position of the patch before computing the kernel
            cuda.syncthreads()
            
            cov_i = cuda.shared.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
            
            DEBUG_E1 = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
            DEBUG_E2 = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
            DEBUG_L = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

            # coordinates of the pixel neigbhour (one for each thread)
            thread_pixel_idx = uint16(patch_center_pos[1]) + tx
            thread_pixel_idy = uint16(patch_center_pos[0]) + ty  
            
            # checking if pixel is r, g or b
            channel = get_channel(thread_pixel_idx, thread_pixel_idy)

            if image_index == 0:
                compute_interpolated_kernel_cov(ref_img, patch_center_pos, cov_i,
                                                k_detail, k_denoise, D_th, D_tr, k_stretch,
                                                k_shrink,DEBUG_E1, DEBUG_E2, DEBUG_L)
                local_r = 1 # for all 9 threads

            else:
                compute_interpolated_kernel_cov(comp_imgs[image_index - 1], patch_center_pos, cov_i,
                                                k_detail, k_denoise, D_th, D_tr, k_stretch,
                                                k_shrink,DEBUG_E1, DEBUG_E2, DEBUG_L)
                
                # robustness
                if 0 <= thread_pixel_idx < input_size_x and 0 <= thread_pixel_idy < input_size_y: # inbound
                    local_r = fetch_robustness(thread_pixel_idx, thread_pixel_idy, image_index - 1, r, channel) # r is a local variable : different for every thread

                

            # We need to wait the calculation of R and kernel
            cuda.syncthreads()
            

            # in bounds conditions
            if (0 <= thread_pixel_idx < input_size_x and
                0 <= thread_pixel_idy < input_size_y):
                if image_index == 0:
                    c = ref_img[thread_pixel_idy, thread_pixel_idx]
                else:
                    c = comp_imgs[image_index - 1, thread_pixel_idy, thread_pixel_idx]


    
                # applying invert transformation and upscaling
                if image_index == 0:
                    fine_sub_pos_x = SCALE * thread_pixel_idx
                    fine_sub_pos_y = SCALE * thread_pixel_idy
                else:
                    fine_sub_pos_x = SCALE * (thread_pixel_idx - local_optical_flow[0])
                    fine_sub_pos_y = SCALE * (thread_pixel_idy - local_optical_flow[1])

                
                dist = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
                dist[0] = (fine_sub_pos_x - output_pixel_idx)
                dist[1] = (fine_sub_pos_y - output_pixel_idy)
                
                # TODO Debugging
                if image_index == 1 :
                    output_img[output_pixel_idy, output_pixel_idx, 13 + 6*(ty+1) + 2*(tx+1)] = dist[0]
                    output_img[output_pixel_idy, output_pixel_idx, 13 + 6*(ty+1) + 2*(tx+1) + 1] = dist[1]
                

                y = max(0, quad_mat_prod(cov_i, dist))
                # y can be slightly negative because of numerical precision.
                # I clamp it to not explode the error with exp
                w = math.exp(-y/2)
                # kernels are estimated on grey levels, so distances have to
                # be downscaled to grey coarse level


                cuda.atomic.add(val, channel, c*w*local_r)
                cuda.atomic.add(acc, channel, w*local_r)
                
                # TODO debugging only
                if image_index == 1 :
                    if tx == 0 and ty == 0:
                        output_img[output_pixel_idy, output_pixel_idx, 3] = DEBUG_E1[0]
                        output_img[output_pixel_idy, output_pixel_idx, 4] = DEBUG_E1[1]
                        output_img[output_pixel_idy, output_pixel_idx, 5] = DEBUG_E2[0]
                        output_img[output_pixel_idy, output_pixel_idx, 6] = DEBUG_E2[1]
                        output_img[output_pixel_idy, output_pixel_idx, 7] = DEBUG_L[0]
                        output_img[output_pixel_idy, output_pixel_idx, 8] = DEBUG_L[1]
                        output_img[output_pixel_idy, output_pixel_idx, 9] = cov_i[0, 0]
                        output_img[output_pixel_idy, output_pixel_idx, 10] = cov_i[0, 1]
                        output_img[output_pixel_idy, output_pixel_idx, 11] = cov_i[1, 0]
                        output_img[output_pixel_idy, output_pixel_idx, 12] = cov_i[1, 1]
                        
                
            
            # We need to wait the accumulation of the 9 pixels before going for
            # the next image, because sharred arrays will be overwritten
            cuda.syncthreads()
        if tx == 0 and ty == 0:
            for chan in range(3):
                output_img[output_pixel_idy, output_pixel_idx, chan] = val[chan]/(acc[chan] + EPSILON) 

                    
    current_time = time()

    accumulate[blockspergrid, threadsperblock](
        ref_img, comp_imgs, alignments, r, output_img)
    cuda.synchronize()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Data merged on GPU side')

    
    merge_result = output_img.copy_to_host()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' - Data returned from GPU')

    return merge_result
