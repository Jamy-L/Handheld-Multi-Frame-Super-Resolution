# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

@author: jamyl
"""

from time import time

from math import floor, ceil, modf, exp
import numpy as np
import cv2
from numba import cuda, float32, float64, int16
from scipy.ndimage import gaussian_filter1d

from .linalg import bicubic_interpolation
from .utils import getTime, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, hann, hamming
from .linalg import solve_2x2, solve_6x6_krylov
    

def lucas_kanade_optical_flow(ref_img, comp_img, pre_alignment, options, params, debug = False):
    """
    Computes the displacement based on a naive implementation of
    Lucas-Kanade optical flow (https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf)
    (The first method). This method is iterated multiple times, given by :
    params['tuning']['kanadeIter']

    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    comp_img : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference
    pre_alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
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
        
    current_time, verbose, verbose_2 = time(), options['verbose'] > 1, options['verbose'] > 2
    n_iter = params['tuning']['kanadeIter']
    tile_size = params['tuning']['tileSize']

    n_images, imsize_y, imsize_x = comp_img.shape

        
    if verbose:
        t1 = time()
        current_time = time()
        print("Estimating Lucas-Kanade's optical flow")
        
    if params["mode"] == "bayer" : 
        # grey level
        ref_img_grey = (ref_img[::2, ::2] + ref_img[1::2, 1::2] + ref_img[::2, 1::2] + ref_img[1::2, ::2])/4
        comp_img_grey = (comp_img[:,::2, ::2] + comp_img[:,1::2, 1::2] + comp_img[:,::2, 1::2] + comp_img[:,1::2, ::2])/4
        pre_alignment = pre_alignment/2 # dividing by 2 because grey image is twice smaller than bayer
        # and blockmatching in bayer mode returnsaignment to bayer scale
        
        # At this point imsize is bayer and tile_size is grey scale
        n_patch_y = 2 * ceil(imsize_y/(2*tile_size)) + 1 
        n_patch_x = 2 * ceil(imsize_x/(2*tile_size)) + 1
        
    else :
        # in non bayer mode, BM returns alignment to grey scale
        ref_img_grey = ref_img # no need to copy now, they will be copied to gpu later.
        comp_img_grey = comp_img
        
        n_patch_y = 2 * ceil(imsize_y/tile_size) + 1 
        n_patch_x = 2 * ceil(imsize_x/tile_size) + 1
        
    if verbose_2:
        current_time = getTime(
            current_time, ' -- Grey Image estimated')

        
    gradsx = np.empty_like(comp_img_grey)
    gradsy = np.empty_like(comp_img_grey)
    
    # Estimating gradients with cv2 sobel filters
    # Data format is very important. default 8uint is bad, because we use floats
    for i in range(n_images):
        gradsx[i] = cv2.Sobel(comp_img_grey[i], cv2.CV_64F, dx=1, dy=0)
        gradsy[i] = cv2.Sobel(comp_img_grey[i], cv2.CV_64F, dx=0, dy=1)

    if verbose_2:
        current_time = getTime(
            current_time, ' -- Gradients estimated')
    
    # translating BM pure tranlsation to affinity model
    alignment = np.zeros((pre_alignment.shape[:-1] + (6,)))
    alignment[:,:,:,4] = pre_alignment[:,:,:,0]
    alignment[:,:,:,5] = pre_alignment[:,:,:,1]
    
    cuda_alignment = cuda.to_device(alignment)
    cuda_ref_img_grey = cuda.to_device(np.ascontiguousarray(ref_img_grey))
    cuda_comp_img_grey = cuda.to_device(np.ascontiguousarray(comp_img_grey))
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


def lucas_kanade_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, options, params, iter_index):
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
    
    tile_size = params['tuning']['tileSize']
        
    EPSILON =  params['epsilon div']
    n_images, imsize_y, imsize_x = comp_img.shape
    
    _, n_patch_y, n_patch_x, _ = alignment.shape
    
    if verbose_2 : 
        print(" -- Lucas-Kanade iteration {}".format(iter_index))
        
        
    @cuda.jit
    def get_new_flow(ref_img, comp_img, gradsx, gradsy, alignment, upscaled_flow):
        
        
        image_index, patch_idy, patch_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        pixel_local_idx = cuda.threadIdx.x # Position relative to the patch
        pixel_local_idy = cuda.threadIdx.y
        
        pixel_global_idx = tile_size//2 * (patch_idx - 1) + pixel_local_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
        pixel_global_idy = tile_size//2 * (patch_idy - 1) + pixel_local_idy
        
        ATA = cuda.shared.array((6,6), dtype = DEFAULT_CUDA_FLOAT_TYPE)
        ATB = cuda.shared.array(6, dtype = DEFAULT_CUDA_FLOAT_TYPE)
        
        # parallel init
        if cuda.threadIdx.x <= 5 and cuda.threadIdx.y <=5 :
            ATA[cuda.threadIdx.y, cuda.threadIdx.x] = 0
        if cuda.threadIdx.y == 6 and cuda.threadIdx.x <= 5 :
                ATB[cuda.threadIdx.x] = 0
  
        
        
        inbound = (0 <= pixel_global_idx < imsize_x) and (0 <= pixel_global_idy < imsize_y)
    
        if inbound :
            # Warp I with W(x; p) to compute I(W(x; p))
            new_idx = ((1+alignment[image_index, patch_idy, patch_idx, 0])*pixel_global_idx +
                          alignment[image_index, patch_idy, patch_idx, 2]*pixel_global_idy +
                          alignment[image_index, patch_idy, patch_idx, 4])
            
            new_idy = (alignment[image_index, patch_idy, patch_idx, 1]*pixel_global_idx +
                          (1+alignment[image_index, patch_idy, patch_idx, 3])*pixel_global_idy +
                              alignment[image_index, patch_idy, patch_idx, 5])
        
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
            gradx = bicubic_interpolation(buffer_val, pos)
            # gradx = gradsx[image_index, new_idy, new_idx]
            
            buffer_val[0, 0] = gradsy[image_index, floor_y, floor_x]
            buffer_val[0, 1] = gradsy[image_index, floor_y, ceil_x]
            buffer_val[1, 0] = gradsy[image_index, ceil_y, floor_x]
            buffer_val[1, 1] = gradsy[image_index, ceil_y, ceil_x]
            grady = bicubic_interpolation(buffer_val, pos)
            # grady = gradsy[image_index, new_idy, new_idx]
            
            buffer_val[0, 0] = comp_img[image_index, floor_y, floor_x]
            buffer_val[0, 1] = comp_img[image_index, floor_y, ceil_x]
            buffer_val[1, 0] = comp_img[image_index, ceil_y, floor_x]
            buffer_val[1, 1] = comp_img[image_index, ceil_y, ceil_x]
            comp_val = bicubic_interpolation(buffer_val, pos)
            gradt = comp_val- ref_img[pixel_global_idy, pixel_global_idx]
            # gradt = comp_img[image_index, new_idy, new_idx] - ref_img[pixel_global_idy, pixel_global_idx]
            
            # exponentially decreasing window, of size tile_size_lk
            r_square = ((pixel_local_idx - tile_size/2)**2 +
                        (pixel_local_idy - tile_size/2)**2)
            sigma_square = (tile_size/2)**2
            w = exp(-r_square/(3*sigma_square)) # TODO this is not improving LK so much... 3 seems to be a good coef though.
            
            cuda.atomic.add(ATB, 0, -pixel_global_idx*gradx*gradt*w)
            cuda.atomic.add(ATB, 1, -pixel_global_idx*grady*gradt*w)
            cuda.atomic.add(ATB, 2, -pixel_global_idy*gradx*gradt*w)
            cuda.atomic.add(ATB, 3, -pixel_global_idy*grady*gradt*w)
            cuda.atomic.add(ATB, 4, -gradx*gradt*w)
            cuda.atomic.add(ATB, 5, -grady*gradt*w)
            
            # Compute the Hessian matrix
            for i in range(6):
                for j in range(6):
                    a = 1
                    if i<= 1:
                        a*= pixel_global_idx
                    elif i<= 3:
                        a*= pixel_global_idy
                    
                    if i%2 == 0:
                        a*=gradx
                    else:
                        a*=grady
                        
                        
                    if j<= 1:
                        a*= pixel_global_idx
                    elif j<= 3:
                        a*= pixel_global_idy
                    
                    if j%2 == 0:
                        a*=gradx
                    else:
                        a*=grady
                    
                    cuda.atomic.add(ATA, (i, j), a*w)
                        
        # TODO is there a clever initialisation for delta p ?
        # Zero init of delta p
        alignment_step = cuda.shared.array(6, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        if cuda.threadIdx.x <= 5 and cuda.threadIdx.y == 0:    
            alignment_step[cuda.threadIdx.x] = 0
            
        cuda.syncthreads()
        solvable = solve_6x6_krylov(ATA, ATB, alignment_step, 7)
        # We cannot know in advance if the system is solvable. If it is not, the 
        # flow is not updated                     
        if solvable and cuda.threadIdx.x <= 5 and cuda.threadIdx.y == 0:
            # No racing condition, one thread for one id
            alignment[image_index, patch_idy, patch_idx, cuda.threadIdx.x] += alignment_step[cuda.threadIdx.x]
        
        cuda.syncthreads()
        if pixel_local_idx == 0 and pixel_local_idy == 0: # single threaded section
            if upscaled_flow : #uspcaling because grey img is 2x smaller
                alignment[image_index, patch_idy, patch_idx, 4] *= 2
                alignment[image_index, patch_idy, patch_idx, 5] *= 2
    
    current_time = time()
    
    get_new_flow[[(n_images, n_patch_y, n_patch_x), (tile_size, tile_size)]
        ](ref_img, comp_img, gradsx, gradsy, alignment, 
            ((n_iter - 1) == iter_index) and (params['mode'] == 'bayer'))   
            # last 2 components (fixed translation) must be doubled during the
            # last iteration in bayer mode, because the real image is twice as big
            # as grey image.
    
    if verbose_2:
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')

    return alignment

@cuda.jit(device=True)
def warp_flow_x(pos, local_flow):
    return local_flow[0]*pos[1] + local_flow[2]*pos[0] + local_flow[4]

@cuda.jit(device=True)
def warp_flow_y(pos, local_flow):
    return local_flow[1]*pos[1] + local_flow[3]*pos[0] + local_flow[5]


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
    pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    pos[0] = round(idy_sub)
    pos[1] = round(idx_sub)
    patch_idy_bottom = round((idy_sub + (tile_size//2))//(tile_size//2))
    patch_idy_top = patch_idy_bottom - 1

    patch_idx_right = round((idx_sub + (tile_size//2))//(tile_size//2))
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
        flow_x = warp_flow_x(pos, optical_flows[-1, -1])
        flow_y = warp_flow_y(pos, optical_flows[-1, -1])
        
    elif patch_idy_top >= imshape[0]:
        if patch_idx_left >= 0 and patch_idx_right <imshape[1]:
            flow_x = (warp_flow_x(pos, optical_flows[-1, patch_idx_left]) + warp_flow_x(pos, optical_flows[-1, patch_idx_right]))/2
            flow_y = (warp_flow_y(pos, optical_flows[-1, patch_idx_left]) + warp_flow_y(pos, optical_flows[-1, patch_idx_right]))/2
        elif patch_idx_left >= 0 :
            flow_x = warp_flow_x(pos, optical_flows[-1, patch_idx_left])
            flow_y = warp_flow_y(pos, optical_flows[-1, patch_idx_left])
        elif patch_idx_right < imshape[1] :
            flow_x = warp_flow_x(pos, optical_flows[-1, patch_idx_right])
            flow_y = warp_flow_y(pos, optical_flows[-1, patch_idx_right])

    elif patch_idx_left >= imshape[1]:
        if patch_idy_top >= 0 and patch_idy_bottom < imshape[0]:
            flow_x = (warp_flow_x(pos, optical_flows[patch_idy_top, -1]) + warp_flow_x(pos, optical_flows[patch_idy_bottom, -1]))/2
            flow_y = (warp_flow_y(pos, optical_flows[patch_idy_top, -1]) + warp_flow_y(pos, optical_flows[patch_idy_bottom, -1]))/2
        elif patch_idy_top >= 0 :
            flow_x = warp_flow_x(pos, optical_flows[patch_idy_top, -1])
            flow_y = warp_flow_y(pos, optical_flows[patch_idy_top, -1])
        elif patch_idy_bottom < imshape[1] :
            flow_x = warp_flow_x(pos, optical_flows[patch_idy_bottom, -1])
            flow_y = warp_flow_y(pos, optical_flows[patch_idy_bottom, -1])
            
            
    # corner conditions
    elif patch_idy_bottom >= imshape[0] and patch_idx_left < 0:
        flow_x = warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right])
        flow_y = warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right])

    elif patch_idy_bottom >= imshape[0] and patch_idx_right >= imshape[1]:
        flow_x = warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left])
        flow_y = warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left])

    elif patch_idy_top < 0 and patch_idx_left < 0:
        flow_x = warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right])
        flow_y = warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right])

    elif patch_idy_top < 0 and patch_idx_right >= imshape[1]:
        flow_x = warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left])
        flow_y = warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left])

    # side conditions
    elif patch_idy_bottom >= imshape[0]:
        flow_x = (warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left]) +
                  warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right]))/2
        flow_y = (warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left]) +
                  warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right]))/2

    elif patch_idy_top < 0:
        flow_x = (warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
                warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right]))/2
        flow_y = (warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
                warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right]))/2

    elif patch_idx_left < 0:
        flow_x = (warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right]) +
                warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right]))/2
        flow_y = (warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right]) +
                warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right]))/2

    elif patch_idx_right >= imshape[1]:
        flow_x = (warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
                warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left]))/2
        flow_y = (warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left]) +
                warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left]))/2

    # general case
    else:
        # Averaging patches with a window function
        tl = hamming(pos[1]%(tile_size/2), pos[0]%(tile_size/2), tile_size)
        tr = hamming(pos[1]%(tile_size/2), (tile_size/2) - pos[0]%(tile_size/2), tile_size)
        bl = hamming((tile_size/2) - pos[1]%(tile_size/2), pos[0]%(tile_size/2), tile_size)
        br = hamming((tile_size/2) - pos[1]%(tile_size/2), (tile_size/2) - pos[0]%(tile_size/2), tile_size)
        
        k = tl + tr + br + bl
        
        
        flow_x = (warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_left])*tl +
                  warp_flow_x(pos, optical_flows[patch_idy_top, patch_idx_right])*tr +
                  warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_left])*bl +
                  warp_flow_x(pos, optical_flows[patch_idy_bottom, patch_idx_right])*br)/k
        flow_y = (warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_left])*tl +
                  warp_flow_y(pos, optical_flows[patch_idy_top, patch_idx_right])*tr +
                  warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_left])*bl +
                  warp_flow_y(pos, optical_flows[patch_idy_bottom, patch_idx_right])*br)/k
    
 
    local_flow[0] = flow_x
    local_flow[1] = flow_y
