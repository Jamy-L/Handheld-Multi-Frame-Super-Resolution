# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:31:38 2022

@author: jamyl
"""
import time
import math

import numpy as np
from numba import cuda
import torch
import torch.nn.functional as F

from .utils import getTime, clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE
from .utils_image import cuda_downsample


def init_block_matching(ref_img, options, params):
    '''
    Returns the pyramid representation of ref_img, that will be used for 
    future block matching

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image
    options : dict
        options.
    params : dict
        parameters.

    Returns
    -------
    referencePyramid : list [device Array]
        pyramid representation of the image

    '''
    # Initialization.
    h, w = ref_img.shape  # height and width should be identical for all images
    
    tileSize = params['tuning']['tileSizes'][0]
    
    # if needed, pad images with zeros so that getTiles contains all image pixels
    paddingPatchesHeight = (tileSize - h % (tileSize)) * (h % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - w % (tileSize)) * (w % (tileSize) != 0)
    # additional zero padding to prevent artifacts on image edges due to overlapped patches in each spatial dimension
    paddingOverlapHeight = paddingOverlapWidth = tileSize // 2
    # combine the two to get the total padding
    paddingTop = paddingOverlapHeight
    paddingBottom = paddingOverlapHeight + paddingPatchesHeight
    paddingLeft = paddingOverlapWidth
    paddingRight = paddingOverlapWidth + paddingPatchesWidth
    
	# pad all images (by mirroring image edges)
	# separate reference and alternate images
    # ref_img_padded = np.pad(ref_img, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric')
    
    th_ref_img = torch.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    th_ref_img_padded = F.pad(th_ref_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')



    # For convenience
    currentTime, verbose = time.perf_counter(), options['verbose'] > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = params['tuning']['factors']


    # construct 4-level coarse-to fine pyramid of the reference

    referencePyramid = hdrplusPyramid(th_ref_img_padded, factors)
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' --- Create ref pyramid')
    
    return referencePyramid


def align_image_block_matching(img, referencePyramid, options, params, debug=False):
    """
    Align the reference image with the img : returns a patchwise flow such that
    for ptaches py, px :
        img[py, px] ~= ref_img[py + alignments[py, px, 1], 
                               px + alignments[py, px, 0]]

    Parameters
    ----------
    img : device Array[imshape_y, imshape_x]
        image to be compared
    referencePyramid : list [device Array]
        Pyramid representation of the ref image
    options : dict
        options.
    params : dict
        parameters.
    debug : Bool, optional
        When True, a list with the alignment at each step is returned. The default is False.

    Returns
    -------
    alignments : device Array[n_patchs_y, n_patchs_x, 2]
        The patch layout is explained in the hdr+ IPOl article

    """
    # Initialization.
    h, w = img.shape  # height and width should be identical for all images
    
    if params['mode'] == 'bayer':
        tileSize = 2 * params['tuning']['tileSizes'][0]
    else:
        tileSize = params['tuning']['tileSizes'][0]
    # if needed, pad images with zeros so that getTiles contains all image pixels
    paddingPatchesHeight = (tileSize - h % (tileSize)) * (h % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - w % (tileSize)) * (w % (tileSize) != 0)
    # additional zero padding to prevent artifacts on image edges due to overlapped patches in each spatial dimension
    paddingOverlapHeight = paddingOverlapWidth = tileSize // 2
    # combine the two to get the total padding
    paddingTop = paddingOverlapHeight
    paddingBottom = paddingOverlapHeight + paddingPatchesHeight
    paddingLeft = paddingOverlapWidth
    paddingRight = paddingOverlapWidth + paddingPatchesWidth
    
	# pad all images (by mirroring image edges)
	# separate reference and alternate images
    
    th_img = torch.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    # img_padded = np.pad(img, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric')
    img_padded = F.pad(th_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')
    

    # For convenience
    currentTime, verbose = time.perf_counter(), options['verbose'] > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = params['tuning']['factors']
    tileSizes = params['tuning']['tileSizes']
    distances = params['tuning']['distances']
    searchRadia = params['tuning']['searchRadia']

    upsamplingFactors = factors[1:] + [1]
    previousTileSizes = tileSizes[1:] + [None]


    # Align alternate image to the reference image

    # 4-level coarse-to fine pyramid of alternate image
    alternatePyramid = hdrplusPyramid(img_padded, factors)
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' --- Create alt pyramid')

    # succesively align from coarsest to finest level of the pyramid
    alignments = None
    if debug:
        debug_list = []
    
    for lv in range(len(referencePyramid)):
        alignments = align_on_a_level(
            referencePyramid[lv],
            alternatePyramid[lv],
            options,
            upsamplingFactors[-lv - 1],
            tileSizes[-lv - 1],
            previousTileSizes[-lv - 1],
            searchRadia[-lv - 1],
            distances[-lv - 1],
            alignments
        )

        if debug:
            debug_list.append(alignments.copy_to_host())
            
        if verbose:
            cuda.synchronize()
            currentTime = getTime(currentTime, ' --- Align pyramid')
    if debug:
        return debug_list
    return alignments


def hdrplusPyramid(image, factors=[1, 2, 4, 4], kernel='gaussian'):
    '''Construct 4-level coarse-to-fine gaussian pyramid
    as described in the HDR+ paper and its supplement (Section 3.2 of the IPOL article).
    Args:
            image: input image (expected to be a grayscale image downsampled from a Bayer raw image)
            factors: [int], dowsampling factors (fine-to-coarse)
            kernel: convolution kernel to apply before downsampling (default: gaussian kernel)'''
    # Start with the finest level computed from the input
    pyramidLevels = [cuda_downsample(image, kernel, factors[0])]
    # pyramidLevels = [downsample(image, kernel, factors[0])]

    # Subsequent pyramid levels are successively created
    # with convolution by a kernel followed by downsampling
    for factor in factors[1:]:
        pyramidLevels.append(cuda_downsample(pyramidLevels[-1], kernel, factor))
        # pyramidLevels.append(downsample(pyramidLevels[-1], kernel, factor))

    # torch to numba, remove batch, channel dimensions
    for i, pyramidLevel in enumerate(pyramidLevels):
        pyramidLevels[i] = cuda.as_cuda_array(pyramidLevel.squeeze())
        
    # Reverse the pyramid to get it coarse-to-fine
    return pyramidLevels[::-1]

def align_on_a_level(referencePyramidLevel, alternatePyramidLevel, options, upsamplingFactor, tileSize, 
                     previousTileSize, searchRadius, distance, previousAlignments):
    """
    Alignment will always be an integer with this function, however it is 
    set to DEFAULT_FLOAT_TYPE. This enables to directly use the outputed
    alignment for ICA without any casting from int to float, which would be hard
    to perform on GPU : Numba is completely powerless and cannot make the
    casting.

    Parameters
    ----------
    referencePyramidLevel : device Array
        DESCRIPTION.
    alternatePyramidLevel : device Array
        DESCRIPTION.
    options : TYPE
        DESCRIPTION.
    upsamplingFactor : int
        DESCRIPTION.
    tileSize : int
        DESCRIPTION.
    previousTileSize : int
        DESCRIPTION.
    searchRadius : int
        DESCRIPTION.
    distance : str
        DESCRIPTION.
    previousAlignments : device Array
        DESCRIPTION.

    Returns
    -------
    upsampledAlignments : TYPE
        DESCRIPTION.

    """
    
    
    # For convenience
    verbose = options['verbose'] > 3
    if verbose :
        cuda.synchronize()
        currentTime = time.perf_counter()
    imshape = referencePyramidLevel.shape
    
    # This formula is checked : it is correct
    # Number of patches that can fit on this level
    h = imshape[0] // (tileSize // 2) - 1
    w = imshape[1] // (tileSize // 2) - 1
    
    # Upsample the previous alignements for initialization
    if previousAlignments is None:
        upsampledAlignments = cuda.to_device(np.zeros((h, w, 2), dtype=DEFAULT_NUMPY_FLOAT_TYPE))
    else:
        # use the upsampled previous alignments as initial guesses
        upsampledAlignments = upsample_alignments(
            referencePyramidLevel,
            alternatePyramidLevel,
            previousAlignments,
            upsamplingFactor,
            tileSize,
            previousTileSize
        )

    # TODO this needs to be modified, it is too slow
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' ---- Upsample alignments')
    
    local_search(referencePyramidLevel, alternatePyramidLevel,
                 tileSize, searchRadius,
                 upsampledAlignments, distance)
    
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' ---- Patchs aligned')
        
    # In the original HDR block matching, supixel precision is obtained here.
    # We do not need that as we use the ICA after block matching

    return upsampledAlignments
    
def upsample_alignments(referencePyramidLevel, alternatePyramidLevel, previousAlignments, upsamplingFactor, tileSize, previousTileSize):
    '''Upsample alignements to adapt them to the next pyramid level (Section 3.2 of the IPOL article).'''
    n_tiles_y_prev, n_tiles_x_prev, _ = previousAlignments.shape
    # Different resolution upsampling factors and tile sizes lead to different vector repetitions
    repeatFactor = upsamplingFactor // (tileSize // previousTileSize)
    # UpsampledAlignments.shape can be less than referencePyramidLevel.shape/tileSize
    # eg when previous alignments could not be computed over the whole image
    n_tiles_y_new = referencePyramidLevel.shape[0] // (tileSize // 2) - 1
    n_tiles_x_new = referencePyramidLevel.shape[1] // (tileSize // 2) - 1


    candidate_flows = cuda.device_array((n_tiles_y_prev*repeatFactor, n_tiles_x_prev*repeatFactor, 3, 2), DEFAULT_NUMPY_FLOAT_TYPE) # 3 candidates
    distances = cuda.device_array((n_tiles_y_prev*repeatFactor, n_tiles_x_prev*repeatFactor, 3), DEFAULT_NUMPY_FLOAT_TYPE) # 3 candidates


    cuda_get_candidate_flows[(n_tiles_x_prev*repeatFactor,
                              n_tiles_y_prev*repeatFactor, 3),
                             (tileSize, tileSize)](
                                 referencePyramidLevel, alternatePyramidLevel, 
                                 previousAlignments, candidate_flows, distances,
                                 upsamplingFactor, repeatFactor, tileSize)

    
    upsampledAlignments = cuda.device_array((n_tiles_y_new, n_tiles_x_new, 2), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    cuda_apply_best_flow[(n_tiles_x_new, n_tiles_y_new), (3)](candidate_flows, distances, repeatFactor, upsampledAlignments)

    return upsampledAlignments
        
        
        
@cuda.jit
def cuda_get_candidate_flows(ref_level, alt_level,
                             prev_al, candidate_al, distances,
                             upsamplingFactor, repeatFactor, new_tileSize):
    ups_tile_x, ups_tile_y, candidate_id = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    
    n_tiles_y_prev, n_tiles_x_prev, _ = prev_al.shape
    
    # position of the new tile within the old tile
    ups_subtile_x = ups_tile_x%repeatFactor
    ups_subtile_y = ups_tile_y%repeatFactor
    
    # computing id for the 3 closest patchs
    if 2 * ups_subtile_x + 1 > repeatFactor:
        x_shift = +1
    else:
        x_shift = -1
        
    if 2 * ups_subtile_y + 1 > repeatFactor:
        y_shift = +1
    else:
        y_shift = -1

    
    # position of the old tile within which the new tile is

    
    # One block dim for each of the 3 candidates
    if candidate_id == 1:
        prev_tile_x = ups_tile_x//repeatFactor + x_shift
        prev_tile_y = ups_tile_y//repeatFactor
    elif candidate_id == 2:
        prev_tile_x = ups_tile_x//repeatFactor
        prev_tile_y = ups_tile_y//repeatFactor + y_shift
    else:
        prev_tile_x = ups_tile_x//repeatFactor
        prev_tile_y = ups_tile_y//repeatFactor
    
    prev_tile_x = clamp(prev_tile_x, 0, n_tiles_x_prev - 1)
    prev_tile_y = clamp(prev_tile_y, 0, n_tiles_y_prev - 1)
    

    # The 3 candidate flow are stored in Global memory
    local_flow = cuda.shared.array(2, DEFAULT_NUMPY_FLOAT_TYPE)


    if tx == 0 and ty <= 1:
        local_flow[ty] = prev_al[prev_tile_y, prev_tile_x, ty] * upsamplingFactor
        candidate_al[ups_tile_y, ups_tile_x, candidate_id, ty] = local_flow[ty]
    cuda.syncthreads()
    

    dist = cuda_upscale_L1_dist(ref_level, alt_level, local_flow, new_tileSize)
    if tx == 0 and ty == 0:
        distances[ups_tile_y, ups_tile_x, candidate_id] = dist
    
    
@cuda.jit(device=True) 
def cuda_upscale_L1_dist(ref_level, alt_level, local_flow, new_tileSize):
    ups_tile_x, ups_tile_y = cuda.blockIdx.x, cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tile_size = cuda.blockDim.y
    
    x = int(ups_tile_x * new_tileSize//2 + tx)
    y = int(ups_tile_y * new_tileSize//2 + ty)

    new_x = int(x + local_flow[0])
    new_y = int(y + local_flow[1])
    
    
    d = cuda.shared.array(32*32, DEFAULT_CUDA_FLOAT_TYPE)
    z = ty*tile_size + tx
    d[z] = abs(ref_level[y, x]-alt_level[new_y, new_x])
    cuda.syncthreads()
    
    # reduction
    N_reduction = int(math.log2(tile_size**2))
    
    step = 1
    for i in range(N_reduction):
        cuda.syncthreads()
        if z%(2*step) == 0:
            d[z] += d[z + step]
        
        step *= 2
    
    cuda.syncthreads()
    return d[0]

@cuda.jit
def cuda_apply_best_flow(candidate_flows, distances, repeatFactor, upsampledAlignments):
    ups_tile_x, ups_tile_y = cuda.blockIdx.x, cuda.blockIdx.y
    candidate_id = cuda.threadIdx.x
    
    # time to group the global mem request. 
    if repeatFactor > 1:
        local_dist = distances[ups_tile_y, ups_tile_x, candidate_id]
        
        # if this condition is met, the flow on this patch cannot be obtained by
        # upsampling because it was not computed previously 
        if ups_tile_x >= distances.shape[1] or ups_tile_y >= distances.shape[0]:
            # the flow is set as 0 by 2 threads
            if cuda.threadIdx.x <= 1:
                upsampledAlignments[ups_tile_y, ups_tile_x, cuda.threadIdx.x] = 0
        else:
            # finding the best of the 3 candidates
            min_dist = cuda.shared.array(1, DEFAULT_CUDA_FLOAT_TYPE)
            if cuda.threadIdx.x == 0:
                min_dist[0] = 1/0 # + infinity
            cuda.syncthreads()
            
            cuda.atomic.min(min_dist, 0, local_dist)
            cuda.syncthreads()
            
            if min_dist[0] == local_dist:
                upsampledAlignments[ups_tile_y, ups_tile_x, 0] = candidate_flows[ups_tile_y, ups_tile_x, candidate_id, 0]
                upsampledAlignments[ups_tile_y, ups_tile_x, 1] = candidate_flows[ups_tile_y, ups_tile_x, candidate_id, 1]

def local_search(referencePyramidLevel, alternatePyramidLevel,
                 tileSize, searchRadius,
                 upsampledAlignments, distance):

    h, w, _ = upsampledAlignments.shape
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(w/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(h/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if distance == 'L1':
        cuda_L1_local_search[blockspergrid, threadsperblock](referencePyramidLevel, alternatePyramidLevel,
                                                      tileSize, searchRadius,
                                                      upsampledAlignments)
    elif distance == 'L2':
        # TODO L2 here
        cuda_L1_local_search[blockspergrid, threadsperblock](referencePyramidLevel, alternatePyramidLevel,
                                                      tileSize, searchRadius,
                                                      upsampledAlignments)
    else:
        raise ValueError('Unknown distance : {}'.format(distance))
        
        
@cuda.jit
def cuda_L1_local_search(referencePyramidLevel, alternatePyramidLevel,
                         tileSize, searchRadius, upsampledAlignments):
    n_patchs_y, n_patchs_x, _ = upsampledAlignments.shape
    h, w = alternatePyramidLevel.shape
    tile_x, tile_y = cuda.grid(2)
    if not(0 <= tile_y < n_patchs_y and
           0 <= tile_x < n_patchs_x):
        return
    
    local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_flow[0] = upsampledAlignments[tile_y, tile_x, 0]
    local_flow[1] = upsampledAlignments[tile_y, tile_x, 1]

    patch_pos_x = tile_x * tileSize//2
    patch_pos_y = tile_y * tileSize//2
    
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = patch_pos_x + j
            idy = patch_pos_y + i
            local_ref[i, j] = referencePyramidLevel[idy, idx]
        
    min_dist = +1/0 #init as infty
    min_shift_x = min_shift_y = 0
    # window search
    for search_shift_y in range(-searchRadius, searchRadius + 1):
        for search_shift_x in range(-searchRadius, searchRadius + 1):
            # computing dist
            dist = 0
            for i in range(tileSize):
                for j in range(tileSize):
                    new_idx = patch_pos_x + j + int(local_flow[0])
                    new_idy = patch_pos_y + i + int(local_flow[1])
                    
                    if (0 <= new_idx < w and
                        0 <= new_idy < h):
                        diff = local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx]
                        dist += abs(diff)
                    else:
                        dist = +1/0
                    
                    
            if dist < min_dist :
                min_dist = dist
                min_shift_y = search_shift_y
                min_shift_x = search_shift_x
    
    upsampledAlignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
    upsampledAlignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y
    
# @cuda.jit
# def cuda_L2_local_search(referencePyramidLevel, alternatePyramidLevel,
#                          tileSize, searchRadius, upsampledAlignments):
#     n_patchs_y, n_patchs_x, _ = upsampledAlignments.shape
#     h, w = alternatePyramidLevel.shape
#     tile_x, tile_y = cuda.grid(2)
#     if not(0 <= tile_y < n_patchs_y and
#            0 <= tile_x < n_patchs_x):
#         return
    
#     local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
#     local_flow[0] = upsampledAlignments[tile_y, tile_x, 0]
#     local_flow[1] = upsampledAlignments[tile_y, tile_x, 1]

#     patch_pos_x = tile_x * tileSize//2
#     patch_pos_y = tile_y * tileSize//2
    
#     local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
#     for i in range(tileSize):
#         for j in range(tileSize):
#             idx = patch_pos_x + j
#             idy = patch_pos_y + i
#             local_ref[i, j] = referencePyramidLevel[idy, idx]
        
#     min_dist = +1/0 #init as infty
#     min_shift_x = min_shift_y = 0
#     # window search
#     for search_shift_y in range(-searchRadius, searchRadius + 1):
#         for search_shift_x in range(-searchRadius, searchRadius + 1):
#             # computing dist
#             dist = 0
#             for i in range(tileSize):
#                 for j in range(tileSize):
#                     new_idx = patch_pos_x + j + int(local_flow[0])
#                     new_idy = patch_pos_y + i + int(local_flow[1])
                    
#                     if (0 <= new_idx < w and
#                         0 <= new_idy < h):
#                         diff = local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx]
#                         dist += diff*diff
#                     else:
#                         dist = +1/0
                    
                    
#             if dist < min_dist :
#                 min_dist = dist
#                 min_shift_y = search_shift_y
#                 min_shift_x = search_shift_x
    
#     upsampledAlignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
#     upsampledAlignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y
    
# def get_patch_distance(referencePyramidLevel, alternatePyramidLevel,
#                        tileSize, searchRadius,
#                        upsampledAlignments, distance):
    
#     sR = 2*searchRadius + 1
#     h, w, _ = upsampledAlignments.shape
#     dst = cuda.device_array((h, w, sR, sR), DEFAULT_NUMPY_FLOAT_TYPE)
#     threadsPerBlock = (tileSize, tileSize)
#     blocks = (w, h, sR*sR)
#     if distance == 'L1':
#         cuda_computeL1Distance_[blocks, threadsPerBlock](referencePyramidLevel, alternatePyramidLevel,
#                                                           tileSize, searchRadius,
#                                                           upsampledAlignments, dst)
#     elif distance == 'L2':
#         cuda_computeL2Distance_[blocks, threadsPerBlock](referencePyramidLevel, alternatePyramidLevel,
#                                                           tileSize, searchRadius,
#                                                           upsampledAlignments, dst)
#     else:
#         raise ValueError('Unknown distance : {}'.format(distance))
#     return dst

# @cuda.jit
# def cuda_computeL1Distance_(referencePyramidLevel, alternatePyramidLevel,
#                             tileSize, searchRadius,
#                             upsampledAlignments, dst):
    
#     tile_x, tile_y, z = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
#     tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
#     sR = 2*searchRadius+1
#     sRx = int(z//sR)
#     sRy = int(z%sR)
    
#     idx = int(tile_x * tileSize//2 + tx)
#     idy = int(tile_y * tileSize//2 + ty)
    
#     local_flow = cuda.shared.array(2, DEFAULT_NUMPY_FLOAT_TYPE)
#     if cuda.threadIdx.x == 0 and cuda.threadIdx.y <= 1:
#         local_flow[cuda.threadIdx.y] = upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
#     cuda.syncthreads()
    
#     new_idx = int(idx + local_flow[0] + sRx - searchRadius)
#     new_idy = int(idy + local_flow[1] + sRy - searchRadius)
    
#     # 32x32 is the max size. We may need less, but shared array size must
#     # be known at compilation time. working with a flattened array makes reduction
#     # easier
#     d = cuda.shared.array((32*32), DEFAULT_CUDA_FLOAT_TYPE)
    
#     z = tx + ty*tileSize # flattened id
#     if not (0 <= new_idx < referencePyramidLevel.shape[1] and
#             0 <= new_idy < referencePyramidLevel.shape[0]) :
#         local_diff = 1/0 # infty out of bound
#     else :
#         local_diff = referencePyramidLevel[idy, idx]-alternatePyramidLevel[new_idy, new_idx]
        
#     d[z] = abs(local_diff)
    
    
#     # sum reduction
#     N_reduction = int(math.log2(tileSize**2))
    
#     step = 1
#     for i in range(N_reduction):
#         cuda.syncthreads()
#         if z%(2*step) == 0:
#             d[z] += d[z + step]
        
#         step *= 2
    
#     cuda.syncthreads()
#     if tx== 0 and ty==0:
#         dst[tile_y, tile_x, sRy, sRx] = d[0]

# @cuda.jit
# def cuda_computeL2Distance_(referencePyramidLevel, alternatePyramidLevel,
#                             tileSize, searchRadius,
#                             upsampledAlignments, dst):
#     tile_x, tile_y, z = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
#     tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
#     sR = 2*searchRadius+1
#     sRx = int(z//sR)
#     sRy = int(z%sR)
    
#     idx = int(tile_x * tileSize//2 + tx)
#     idy = int(tile_y * tileSize//2 + ty)
    
#     local_flow = cuda.shared.array(2, DEFAULT_NUMPY_FLOAT_TYPE)
#     if cuda.threadIdx.x == 0 and cuda.threadIdx.y <= 1:
#         local_flow[cuda.threadIdx.y] = upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
#     cuda.syncthreads()
    
#     new_idx = int(idx + local_flow[0] + sRx - searchRadius)
#     new_idy = int(idy + local_flow[1] + sRy - searchRadius)
    
    
#     # 32x32 is the max size. We may need less, but shared array size must
#     # be known at compilation time. working with a flattened array makes reduction
#     # easier
#     d = cuda.shared.array((32*32), DEFAULT_CUDA_FLOAT_TYPE)
    
#     z = tx + ty*tileSize # flattened id
#     if not (0 <= new_idx < referencePyramidLevel.shape[1] and
#             0 <= new_idy < referencePyramidLevel.shape[0]) :
#         local_diff = 1/0 # infty out of bound
#     else :
#         local_diff = referencePyramidLevel[idy, idx] - alternatePyramidLevel[new_idy, new_idx]
        
#     d[z] = local_diff*local_diff
    
    
#     # sum reduction
#     N_reduction = int(math.log2(tileSize**2))
    
#     step = 1
#     for i in range(N_reduction):
#         cuda.syncthreads()
#         if z%(2*step) == 0:
#             d[z] += d[z + step]
        
#         step *= 2
    
#     cuda.syncthreads()
#     if tx== 0 and ty==0:
#         dst[tile_y, tile_x, sRy, sRx] = d[0]
    
# @cuda.jit
# def cuda_minimize_distance(distances, alignment, searchRadius):
#     tile_x, tile_y = cuda.blockIdx.x, cuda.blockIdx.y
#     tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    
#     # grouping glob memory call into a single big request
#     local_dist = distances[tile_y, tile_x, ty, tx]
    
#     min_dist = cuda.shared.array(1, DEFAULT_CUDA_FLOAT_TYPE)
#     if tx == 0 and ty ==0:
#         min_dist[0] = 1/0 # infinity
#     cuda.syncthreads()
    
#     cuda.atomic.min(min_dist, 0, local_dist)
    
#     cuda.syncthreads()
    
#     # there is technically a racing condition in the very rare case
#     # where 2 patches have exactly the same distance.
#     if local_dist == min_dist[0]:
#         alignment[tile_y, tile_x, 1] += ty - searchRadius
#         alignment[tile_y, tile_x, 0] += tx - searchRadius