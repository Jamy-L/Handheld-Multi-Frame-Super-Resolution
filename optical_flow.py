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

def lucas_kanade_optical_flow(ref_img_bayer, comp_img_bayer, pre_alignment_bayer, options, params):
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

    Returns
    -------
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        alignment vector for each tile of each image

    """
    current_time, verbose = time(), options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']

    n_images, n_patch_y, n_patch_x, _ \
        = pre_alignment_bayer.shape
        
    if verbose:
        t1 = time()
        current_time = time()
        print("Estimating Lucas-Kanade's optical flow")

    # grey level
    ref_img = ref_img_bayer[::2, ::2]/3 + ref_img_bayer[1::2, 1::2]/3 + ref_img_bayer[::2, 1::2]/6 + ref_img_bayer[1::2, ::2]/6
    comp_img = comp_img_bayer[:,::2, ::2]/3 + comp_img_bayer[:,1::2, 1::2]/3 + comp_img_bayer[:,::2, 1::2]/6 + comp_img_bayer[:,1::2, ::2]/6
    pre_alignment = pre_alignment_bayer/2 # dividing by 2 because grey image is decimated by 2

    if verbose:
        current_time = getTime(
            current_time, ' -- Grey Image decimated')

        
    gradsx = np.empty_like(comp_img)
    gradsy = np.empty_like(comp_img)
    
    for image_id in range(n_images):
        
        # Estimating gradients with cv2 sobel filters
        # Data format is very important. default 8uint is bad, because we use negative
        # Values, and because may go up to a value of 3080. We need signed int16
        gradsx[image_id] = cv2.Sobel(comp_img[image_id], cv2.CV_16S, dx=1, dy=0)
        gradsy[image_id] = cv2.Sobel(comp_img[image_id], cv2.CV_16S, dx=0, dy=1)

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
        alignment = lucas_kanade_optical_flow_iteration(
            cuda_ref_img, cuda_gradsx, cuda_gradsy, cuda_comp_img, alignment,
            options, params, iter_index)
    
    if verbose:
        getTime(t1, 'Flows estimated')
        print('\n')
    
    return alignment


def lucas_kanade_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, options, params, iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img : Array [imsize_y, imsize_x]
        Ref image
    gradsx : Array [n_images, imsize_y, imsize_x]
        Horizontal gradient of the compared images
    gradsy : Array [n_images, imsize_y, imsize_x]
        Vertical gradient of thecompared images
    comp_img : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The inial alignment of the tile 
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
            new_idx = round(pixel_global_idx + alignment[image_index, pixel_local_idy, pixel_local_idx, 0])
            new_idy = round(pixel_global_idy + alignment[image_index, pixel_local_idy, pixel_local_idx, 1])
        
        inbound = inbound and (0 <= new_idx < imsize_x) and (0 <= new_idy < imsize_y)
        
        if inbound:
            gradx = gradsx[image_index, new_idy, new_idx]
            grady = gradsy[image_index, new_idy, new_idx]
            gradt = int16(comp_img[image_index, new_idy, new_idx]) - int16(ref_img[new_idy, new_idx])
            
            cuda.atomic.add(ATB, 0, gradx*gradt)
            cuda.atomic.add(ATB, 1, grady*gradt)
            
            cuda.atomic.add(ATA, (0, 0), gradx**2)
            cuda.atomic.add(ATA, (1, 0), gradx*grady)
            cuda.atomic.add(ATA, (0, 1), gradx*grady)
            cuda.atomic.add(ATA, (1, 1), grady**2)
        
        # We need the entire tile accumulation
        tile_disp = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
        cuda.syncthreads()
        if pixel_local_idx == 0 and pixel_local_idy == 0 and inbound:
            solve_2x2(ATA, ATB, tile_disp) 
            alignment[image_index, patch_idy, patch_idx, 0] += tile_disp[0]
            alignment[image_index, patch_idy, patch_idx, 1] += tile_disp[1]
            
            if last_it : #uspcaling because grey img is 2x smaller
                alignment[image_index, patch_idy, patch_idx, 0] *= 2
                alignment[image_index, patch_idy, patch_idx, 0] *= 2
    
    current_time = time()
    
    get_new_flow[[(n_images, n_patch_y, n_patch_x), (tile_size, tile_size)]
        ](ref_img, comp_img, gradsx, gradsy, alignment, n_iter - 1 == iter_index)
    
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
    # out of bounds. With zero flow, they will be discarded later
    if (idx_sub < 0 or idx_sub >= imsize[1] or
        idy_sub < 0 or idy_sub >= imsize[0]):
        flow_x = 0
        flow_y = 0

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

# %% test code, to remove in the final version

# raw_ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N000.dng'
#                         )
# ref_img = raw_ref_img.raw_image.copy()


# comp_images = rawpy.imread(
#     'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
# for i in range(2, 10):
#     comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
#                                                       ).raw_image.copy()[None], axis=0)

# pre_alignment = np.load(
#     'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')

# n_images, n_patch_y, n_patch_x, _ = pre_alignment.shape
# tile_size = int(32/2)
# native_im_size = ref_img.shape

# params = {'tuning': {'tileSizes': tile_size}, 'scale': 1}
# params['tuning']['kanadeIter'] = 3





# #%%
# t1 = time()
# final_alignments = lucas_kanade_optical_flow(
#     ref_img, comp_images, pre_alignment, {"verbose": 3}, params)
# print('\nTotal ellapsed time : ', time() - t1)

# a,b = final_alignments[0, :, :]*2, pre_alignment[0, :, :]*2