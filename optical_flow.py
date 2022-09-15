# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

@author: jamyl
"""
import matplotlib
import matplotlib.pyplot as plt
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.genericUtils import getTime
from linalg import solve_2x2

import cv2
import numpy as np
from time import time
import cupy as cp
import rawpy
from tqdm import tqdm
from numba import cuda, float32, float64, int16

DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

def lucas_kanade_optical_flow(ref_img_bayer, comp_img_bayer, pre_alignment_bayer, options, params, debug = False):
    """
    Computes the displacement based on a naive implementation of
    Lucas-Kanade optical flow (https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf)
    (The first method). This method is iterated multiple times, given by :
    params['tuning']['kanadeIter']

    Parameters
    ----------
    ref_img_Bayer : Array[imsize_y, imsize_x]
        The reference image
    comp_img_Bayer : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference
    pre_alignment_bayer : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The alignment vectors obtained by the coarse to fine pyramid search.
    options : Dict
        Options to pass
    params : Dict
        parameters
    debug : When True, returns a list of the alignments of every Lucas-Kanade iteration

    Returns
    -------
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        alignment vector for each tile of each image

    """
    if debug : 
        debug_list = []
        
    current_time, verbose = time(), options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']

    n_images, n_patch_y, n_patch_x, _ \
        = pre_alignment_bayer.shape
        
    if verbose:
        t1 = time()
        current_time = time()
        print("Estimating Lucas-Kanade's optical flow")

    # grey level
    ref_img = (ref_img_bayer[::2, ::2] + ref_img_bayer[1::2, 1::2] + ref_img_bayer[::2, 1::2] + ref_img_bayer[1::2, ::2])/4
    comp_img = (comp_img_bayer[:,::2, ::2] + comp_img_bayer[:,1::2, 1::2] + comp_img_bayer[:,::2, 1::2] + comp_img_bayer[:,1::2, ::2])/4
    pre_alignment = pre_alignment_bayer/2 # dividing by 2 because grey image twice smaller

    if verbose:
        current_time = getTime(
            current_time, ' -- Grey Image estimated')

        
    gradsx = np.empty_like(comp_img)
    gradsy = np.empty_like(comp_img)
    
    # Estimating gradients with cv2 sobel filters
    # Data format is very important. default 8uint is bad, because we use negative
    # Values, and because may go higher. We need signed int16
    for i in range(n_images):
        gradsx[i] = cv2.Sobel(comp_img[i], cv2.CV_16S, dx=1, dy=0)
        gradsy[i] = cv2.Sobel(comp_img[i], cv2.CV_16S, dx=0, dy=1)

    if verbose:
        current_time = getTime(
            current_time, ' -- Gradients estimated')

    alignment = cuda.to_device(pre_alignment) # init of flow, directly on GPU
    cuda_ref_img = cuda.to_device(np.ascontiguousarray(ref_img))
    cuda_comp_img = cuda.to_device(np.ascontiguousarray(comp_img))
    cuda_gradsx = cuda.to_device(gradsx)
    cuda_gradsy = cuda.to_device(gradsy)
    
    current_time = getTime(
        current_time, ' -- Arrays moved to GPU')
    
    for iter_index in range(n_iter):
        lucas_kanade_optical_flow_iteration(
            cuda_ref_img, cuda_gradsx, cuda_gradsy, cuda_comp_img, alignment,
            options, params, iter_index)
        if debug :
            debug_list.append(alignment.copy_to_host())
    
    if verbose:
        getTime(t1, 'Flows estimated')
        print('\n')
    if debug:
        return debug_list
    return alignment


def lucas_kanade_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, options, params, iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img : Array [imsize_y, imsize_x]
        Ref image
    gradx : Array [imsize_y, imsize_x]
        Horizontal gradient of the ref image
    grady : Array [imsize_y, imsize_x]
        Vertical gradient of the ref image
    comp_img : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference
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
    verbose = options['verbose']
    n_iter = params['tuning']['kanadeIter']
    tile_size = int(params['tuning']['tileSizes']/2) #grey tiles are twice as small
    EPSILON =  params['epsilon div']
    n_images, n_patch_y, n_patch_x, _ \
        = alignment.shape
    _, imsize_y, imsize_x = comp_img.shape

    print(" -- Lucas-Kanade iteration {}".format(iter_index))


        
    @cuda.jit
    def get_new_flow(ref_img, comp_img, gradsx, gradsy, alignment, last_it):
        
        
        image_index, patch_idy, patch_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        pixel_local_idx = cuda.threadIdx.x
        pixel_local_idy = cuda.threadIdx.y
        
        inbound = (0 <= pixel_local_idx < tile_size) and (0 <= pixel_local_idy < tile_size)
        
        pixel_global_idx = tile_size//2 * patch_idx + pixel_local_idx
        pixel_global_idy = tile_size//2 * patch_idy + pixel_local_idy
        
        ATA = cuda.shared.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        ATB = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
        ATA[0, 0] = 0
        ATA[0, 1] = 0
        ATA[1, 0] = 0
        ATA[1, 1] = 0
        ATB[0] = 0
        ATB[1] = 0
        
        inbound = inbound and  (0 <= pixel_global_idx < imsize_x) and (0 <= pixel_global_idy < imsize_y)
    
        if inbound :
            # Warp I with W(x; p) to compute I(W(x; p))
            new_idx = round(pixel_global_idx + alignment[image_index, patch_idy, patch_idx, 0])
            new_idy = round(pixel_global_idy + alignment[image_index, patch_idy, patch_idx, 1])
        
        inbound = inbound and (0 <= new_idx < imsize_x) and (0 <= new_idy < imsize_y)
        
        if inbound:
            # Warp the gradient of I with W(x; p)
            gradx = gradsx[image_index, new_idy, new_idx]
            grady = gradsy[image_index, new_idy, new_idx]
            # Images are of unsigned type for speeding up transfers. We need to sign them before substracting
            gradt = int16(comp_img[image_index, new_idy, new_idx]) - int16(ref_img[pixel_global_idy, pixel_global_idx])
            
            cuda.atomic.add(ATB, 0, -gradx*gradt)
            cuda.atomic.add(ATB, 1, -grady*gradt)
            
            # Compute the Hessian matrix
            cuda.atomic.add(ATA, (0, 0), gradx**2)
            cuda.atomic.add(ATA, (1, 0), gradx*grady)
            cuda.atomic.add(ATA, (0, 1), gradx*grady)
            cuda.atomic.add(ATA, (1, 1), grady**2)
        
        # We need the entire tile accumulation
        tile_disp = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cuda.syncthreads()
        if pixel_local_idx == 0 and pixel_local_idy == 0: # single threaded section
            if ATA[0, 0]*ATA[1, 1] - ATA[0, 1]*ATA[1, 0] < EPSILON : # Array cannot be inverted
                tile_disp[0] = 0
                tile_disp[1] = 0
            else:
                solve_2x2(ATA, ATB, tile_disp) 
            alignment[image_index, patch_idy, patch_idx, 0] += tile_disp[0]
            alignment[image_index, patch_idy, patch_idx, 1] += tile_disp[1]
            
            if last_it : #uspcaling because grey img is 2x smaller
                alignment[image_index, patch_idy, patch_idx, 0] *= 2
                alignment[image_index, patch_idy, patch_idx, 1] *= 2
    
    current_time = time()
    
    get_new_flow[[(n_images, n_patch_y, n_patch_x), (tile_size, tile_size)]
        ](ref_img, comp_img, gradsx, gradsy, alignment, (n_iter - 1) == iter_index)
    
    if verbose:
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')

    return alignment


@cuda.jit(device=True)
def get_closest_flow(idx_sub, idy_sub, optical_flows, tile_size, imsize, local_flow):
    """
    Returns the estimated optical flow for a subpixel (or a pixel), based on
    the tile based estimated optical flow.

    Parameters
    ----------
    idx_sub, idy_sub : float
        subpixel ids where the flow must be estimated
    optical_flow : Array[n_tiles_y, n_tiles_x, 2]
        tile based optical flow


    Returns
    -------
    flow : array[2]
        optical flow

    """
    patch_idy_bottom = int(idy_sub//(tile_size//2))
    patch_idy_top = patch_idy_bottom - 1

    patch_idx_right = int(idx_sub//(tile_size//2))
    patch_idx_left = patch_idx_right - 1

    imshape = optical_flows.shape[:2]
    # Index out of bound. With zero flow, they will be discarded later
    if (idx_sub < 0 or idx_sub >= imsize[1] or
        idy_sub < 0 or idy_sub >= imsize[0]):
        flow_x = 0
        flow_y = 0
    
    # Side condition when the required patch is not existing (because it would be 
    # too small for block matching)
    elif patch_idy_top >= imshape[0] and patch_idx_left >= imshape[1]:
        flow_x = optical_flows[-1, -1, 0]
        flow_y = optical_flows[-1, -1, 1]
        
    elif patch_idy_top >= imshape[0]:
        if patch_idx_left >= 0 and patch_idx_right <imshape[1]:
            flow_x = (optical_flows[-1, patch_idx_left, 0] + optical_flows[-1, patch_idx_right, 0])/2
            flow_y = (optical_flows[-1, patch_idx_left, 1] + optical_flows[-1, patch_idx_right, 1])/2
        elif patch_idx_left >= 0 :
            flow_x = optical_flows[-1, patch_idx_left, 0]
            flow_y = optical_flows[-1, patch_idx_left, 1]
        elif patch_idx_right < imshape[1] :
            flow_x = optical_flows[-1, patch_idx_right, 0]
            flow_y = optical_flows[-1, patch_idx_right, 1]

    elif patch_idx_left >= imshape[1]:
        if patch_idy_top >= 0 and patch_idy_bottom < imshape[0]:
            flow_x = (optical_flows[patch_idy_top, -1, 0] + optical_flows[patch_idy_bottom, -1, 0])/2
            flow_y = (optical_flows[patch_idy_top, -1, 1] + optical_flows[patch_idy_bottom, -1, 1])/2
        elif patch_idy_top >= 0 :
            flow_x = optical_flows[patch_idy_top, -1, 0]
            flow_y = optical_flows[patch_idy_top, -1, 1]
        elif patch_idy_bottom < imshape[1] :
            flow_x = optical_flows[patch_idy_bottom, -1, 0]
            flow_y = optical_flows[patch_idy_bottom, -1, 1]
            
            
    # corner conditions
    elif patch_idy_bottom >= imshape[0] and patch_idx_left < 0:
        flow_x = optical_flows[patch_idy_top, patch_idx_right, 0]
        flow_y = optical_flows[patch_idy_top, patch_idx_right, 1]

    elif patch_idy_bottom >= imshape[0] and patch_idx_right >= imshape[1]:
        flow_x = optical_flows[patch_idy_top, patch_idx_left, 0]
        flow_y = optical_flows[patch_idy_top, patch_idx_left, 1]

    elif patch_idy_top < 0 and patch_idx_left < 0:
        flow_x = optical_flows[patch_idy_bottom, patch_idx_right, 0]
        flow_y = optical_flows[patch_idy_bottom, patch_idx_right, 1]

    elif patch_idy_top < 0 and patch_idx_right >= imshape[1]:
        flow_x = optical_flows[patch_idy_bottom, patch_idx_left, 0]
        flow_y = optical_flows[patch_idy_bottom, patch_idx_left, 1]

    # side conditions
    elif patch_idy_bottom >= imshape[0]:
        flow_x = (optical_flows[patch_idy_top, patch_idx_left, 0] +
                optical_flows[patch_idy_top, patch_idx_right, 0])/2
        flow_y = (optical_flows[patch_idy_top, patch_idx_left, 1] +
                optical_flows[patch_idy_top, patch_idx_right, 1])/2

    elif patch_idy_top < 0:
        flow_x = (optical_flows[patch_idy_bottom, patch_idx_left, 0] +
                optical_flows[patch_idy_bottom, patch_idx_right, 0])/2
        flow_y = (optical_flows[patch_idy_bottom, patch_idx_left, 1] +
                optical_flows[patch_idy_bottom, patch_idx_right, 1])/2

    elif patch_idx_left < 0:
        flow_x = (optical_flows[patch_idy_bottom, patch_idx_right, 0] +
                optical_flows[patch_idy_top, patch_idx_right, 0])/2
        flow_y = (optical_flows[patch_idy_bottom, patch_idx_right, 1] +
                optical_flows[patch_idy_top, patch_idx_right, 1])/2

    elif patch_idx_right >= imshape[1]:
        flow_x = (optical_flows[patch_idy_bottom, patch_idx_left, 0] +
                optical_flows[patch_idy_top, patch_idx_left, 0])/2
        flow_y = (optical_flows[patch_idy_bottom, patch_idx_left, 1] +
                optical_flows[patch_idy_top, patch_idx_left, 1])/2

    # general case
    else:
        # Averaging patches
        flow_x = (optical_flows[patch_idy_top, patch_idx_left, 0] +
                optical_flows[patch_idy_top, patch_idx_right, 0] +
                optical_flows[patch_idy_bottom, patch_idx_left, 0] +
                optical_flows[patch_idy_bottom, patch_idx_right, 0])/4
        flow_y = (optical_flows[patch_idy_top, patch_idx_left, 1] +
                optical_flows[patch_idy_top, patch_idx_right, 1] +
                optical_flows[patch_idy_bottom, patch_idx_left, 1] +
                optical_flows[patch_idy_bottom, patch_idx_right, 1])/4
    
 
    local_flow[0] = flow_x
    local_flow[1] = flow_y

