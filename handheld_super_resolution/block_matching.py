# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:31:38 2022

This script contains all the operations corresponding to the function
"MultiScaleBlockMatching" called in Alg. 2: Registration. The pyramid
representations are created and patches are aligned with the block matching
method. 


@author: jamyl
"""
import time
import math

import numpy as np
from numba import cuda
import torch
import torch.nn.functional as F

from .utils import getTime, clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS
from .utils_image import cuda_downsample


def init_block_matching(ref_img, config):
    '''
    Returns the pyramid representation of ref_img, that will be used for 
    future block matching

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
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

    tileSize = config.block_matching.tuning.tile_size

    # if needed, pad images with zeros so that getTiles contains all image pixels
    paddingPatchesHeight = (tileSize - h % (tileSize)) * (h % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - w % (tileSize)) * (w % (tileSize) != 0)

    # combine the two to get the total padding
    paddingTop = 0
    paddingBottom = paddingPatchesHeight
    paddingLeft = 0
    paddingRight = paddingPatchesWidth
    
	# pad all images (by mirroring image edges)
	# separate reference and alternate images
    # ref_img_padded = np.pad(ref_img, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric')
    
    th_ref_img = torch.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    th_ref_img_padded = F.pad(th_ref_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')



    # For convenience
    currentTime, verbose = time.perf_counter(), config.verbose > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = config.block_matching.tuning.factors


    # construct 4-level coarse-to fine pyramid of the reference

    referencePyramid = hdrplusPyramid(th_ref_img_padded, factors)
    if verbose:
        currentTime = getTime(currentTime, ' --- Create ref pyramid')
    
    return referencePyramid


def align_image_block_matching(img, referencePyramid, config, debug=False):
    """
    Align the reference image with the img : returns a patchwise flow such that
    for patches py, px :
        img[py, px] ~= ref_img[py + alignments[py, px, 1], 
                               px + alignments[py, px, 0]]

    Parameters
    ----------
    img : device Array[imshape_y, imshape_x]
        Image to be compared J_i (i>1)
    referencePyramid : list [device Array]
        Pyramid representation of the ref image J_1
    options : dict
        options.
    params : dict
        parameters.
    debug : Bool, optional
        When True, a list with the alignment at each step is returned. The default is False.

    Returns
    -------
    alignments : device Array[n_patchs_y, n_patchs_x, 2]
        Patchwise flow : V_n(p) for each patch (p)

    """
    # Initialization.
    h, w = img.shape  # height and width should be identical for all images
    
    tileSize = config.block_matching.tuning.tile_size
    # if needed, pad images with zeros so that getTiles contains all image pixels
    paddingPatchesHeight = (tileSize - h % (tileSize)) * (h % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - w % (tileSize)) * (w % (tileSize) != 0)

    # combine the two to get the total padding
    paddingTop = 0
    paddingBottom = paddingPatchesHeight
    paddingLeft = 0
    paddingRight = paddingPatchesWidth
    
	# pad all images (by mirroring image edges)
	# separate reference and alternate images
    
    th_img = torch.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    

    img_padded = F.pad(th_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')
    

    # For convenience
    currentTime, verbose = time.perf_counter(), config.verbose > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = config.block_matching.tuning.factors
    tileSizes = config.block_matching.tuning.tile_sizes
    distances = config.block_matching.tuning.metrics
    searchRadia = config.block_matching.tuning.search_radii

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
            config,
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

def align_on_a_level(referencePyramidLevel, alternatePyramidLevel, config, upsamplingFactor, tileSize, 
                     previousTileSize, searchRadius, distance, previousAlignments):
    """
    Alignment will always be an integer with this function, however it is 
    set to DEFAULT_FLOAT_TYPE. This enables to directly use the outputed
    alignment for ICA without any casting from int to float, which would be hard
    to perform on GPU : Numba is completely powerless and cannot make the
    casting.

    """
    
    
    # For convenience
    verbose = config.verbose > 3
    if verbose :
        cuda.synchronize()
        currentTime = time.perf_counter()
    imshape = referencePyramidLevel.shape
    
    # This formula is checked : it is correct
    # Number of patches that can fit on this level
    h = imshape[0] // tileSize
    w = imshape[1] // tileSize
    
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

    # UpsampledAlignments.shape can be less than referencePyramidLevel.shape/tileSize
    # eg when previous alignments could not be computed over the whole image
    n_tiles_y_new = referencePyramidLevel.shape[0] // tileSize
    n_tiles_x_new = referencePyramidLevel.shape[1] // tileSize

    upsampledAlignments = cuda.device_array((n_tiles_y_new, n_tiles_x_new, 2), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(n_tiles_x_new/threadsperblock[1])
    blockspergrid_y = math.ceil(n_tiles_y_new/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_upsample_alignments[blockspergrid, threadsperblock](
        referencePyramidLevel, alternatePyramidLevel,
        upsampledAlignments, previousAlignments,
        upsamplingFactor, tileSize, previousTileSize)

    return upsampledAlignments
        
@cuda.jit
def cuda_upsample_alignments(referencePyramidLevel, alternatePyramidLevel, upsampledAlignments, previousAlignments, upsamplingFactor, tileSize, previousTileSize):
    subtile_x, subtile_y = cuda.grid(2)
    n_tiles_y_prev, n_tiles_x_prev, _ = previousAlignments.shape
    n_tiles_y_new, n_tiles_x_new, _ = upsampledAlignments.shape
    h, w = referencePyramidLevel.shape

    repeatFactor = upsamplingFactor // (tileSize // previousTileSize)
    if not(0 <= subtile_x < n_tiles_x_new and
           0 <= subtile_y < n_tiles_y_new):
        return
    
    # the new subtile is on the side of the image, and is not contained within a bigger old tile
    if (subtile_x >= repeatFactor*n_tiles_x_prev or
        subtile_y >= repeatFactor*n_tiles_y_prev):
        upsampledAlignments[subtile_y, subtile_x, 0] = 0
        upsampledAlignments[subtile_y, subtile_x, 1] = 0
        return
    
    # else
    prev_tile_x = subtile_x//repeatFactor
    prev_tile_y = subtile_y//repeatFactor
    
    candidate_alignment_0_shift = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    candidate_alignment_0_shift[0] = previousAlignments[prev_tile_y, prev_tile_x, 0] * upsamplingFactor
    candidate_alignment_0_shift[1] = previousAlignments[prev_tile_y, prev_tile_x, 1] * upsamplingFactor
    
    # position of the top left pixel in the subtile
    subtile_pos_y = subtile_y*tileSize
    subtile_pos_x = subtile_x*tileSize
    
    # copying ref patch into local memory, because it needs to be read 3 times
    # this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = subtile_pos_x + j
            idy = subtile_pos_y + i            
            local_ref[i, j] = referencePyramidLevel[idy, idx]
    
    
    # position of the new tile within the old tile
    ups_subtile_x = subtile_x%repeatFactor
    ups_subtile_y = subtile_y%repeatFactor
    
    # computing id for the 3 closest patchs
    if 2 * ups_subtile_x + 1 > repeatFactor:
        x_shift = +1
    else:
        x_shift = -1
        
    if 2 * ups_subtile_y + 1 > repeatFactor:
        y_shift = +1
    else:
        y_shift = -1
    
    # 3 Candidates alignments are fetched (by fetching them as early as possible, we may received 
    # them from global memory before we even require them, as calculations are performed during this delay)
    candidate_alignment_vert_shift = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    candidate_alignment_vert_shift[0] = previousAlignments[clamp(prev_tile_y + y_shift, 0, n_tiles_y_prev - 1),
                                                           prev_tile_x,
                                                           0] * upsamplingFactor
    candidate_alignment_vert_shift[1] = previousAlignments[clamp(prev_tile_y + y_shift, 0, n_tiles_y_prev - 1),
                                                           prev_tile_x,
                                                           1] * upsamplingFactor
    
    candidate_alignment_horizontal_shift = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    candidate_alignment_horizontal_shift[0] = previousAlignments[prev_tile_y,
                                                                 clamp(prev_tile_x + x_shift, 0, n_tiles_x_prev - 1),
                                                                 0] * upsamplingFactor
    candidate_alignment_horizontal_shift[1] = previousAlignments[prev_tile_y,
                                                                 clamp(prev_tile_x + x_shift, 0, n_tiles_x_prev - 1),
                                                                 1] * upsamplingFactor
    
    # Choosing the best of the 3 alignments by minimising L1 dist
    dist = +1/0
    optimal_flow_x = 0
    optimal_flow_y = 0
    
    # 0 shift
    dist_ = 0
    for i in range(tileSize):
        for j in range(tileSize):
            new_idy = subtile_pos_y + i + int(candidate_alignment_0_shift[1])
            new_idx = subtile_pos_x + j + int(candidate_alignment_0_shift[0])
            if (0 <= new_idx < w and
                0 <= new_idy < h):
                dist_ += abs(local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx])
            else:
                dist_ = 1/0
    if dist_ < dist:
        dist = dist_
        optimal_flow_x = candidate_alignment_0_shift[0]
        optimal_flow_y = candidate_alignment_0_shift[1]
        
    # vertical shift
    dist_ = 0
    for i in range(tileSize):
        for j in range(tileSize):
            new_idy = subtile_pos_y + i + int(candidate_alignment_vert_shift[1])
            new_idx = subtile_pos_x + j + int(candidate_alignment_vert_shift[0])
            if (0 <= new_idx < w and
                0 <= new_idy < h):
                dist_ += abs(local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx])
            else:
                dist_ = 1/0
    if dist_ < dist:
        dist = dist_
        optimal_flow_x = candidate_alignment_vert_shift[0]
        optimal_flow_y = candidate_alignment_vert_shift[1]
            
    # horizontal shift
    dist_ = 0
    for i in range(tileSize):
        for j in range(tileSize):
            new_idy = subtile_pos_y + i + int(candidate_alignment_horizontal_shift[1])
            new_idx = subtile_pos_x + j + int(candidate_alignment_horizontal_shift[0])
            if (0 <= new_idx < w and
                0 <= new_idy < h):
                dist_ += abs(local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx])
            else:
                dist_ = 1/0
    if dist_ < dist:
        dist = dist_
        optimal_flow_x = candidate_alignment_horizontal_shift[0]
        optimal_flow_y = candidate_alignment_horizontal_shift[1]
    
    # applying best flow
    upsampledAlignments[subtile_y, subtile_x, 0] = optimal_flow_x
    upsampledAlignments[subtile_y, subtile_x, 1] = optimal_flow_y



def local_search(referencePyramidLevel, alternatePyramidLevel,
                 tileSize, searchRadius,
                 upsampledAlignments, distance):

    h, w, _ = upsampledAlignments.shape
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(w/threadsperblock[1])
    blockspergrid_y = math.ceil(h/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if distance == 'L1':
        cuda_L1_local_search[blockspergrid, threadsperblock](referencePyramidLevel, alternatePyramidLevel,
                                                      tileSize, searchRadius,
                                                      upsampledAlignments)
    elif distance == 'L2':
        cuda_L2_local_search[blockspergrid, threadsperblock](referencePyramidLevel, alternatePyramidLevel,
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

    # position of the pixel in the top left corner of the patch
    patch_pos_x = tile_x * tileSize
    patch_pos_y = tile_y * tileSize
    
    # this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = patch_pos_x + j
            idy = patch_pos_y + i
            local_ref[i, j] = referencePyramidLevel[idy, idx]
        
    min_dist = +1/0 #init as infty
    min_shift_y = 0
    min_shift_x = 0
    # window search
    for search_shift_y in range(-searchRadius, searchRadius + 1):
        for search_shift_x in range(-searchRadius, searchRadius + 1):
            # computing dist
            dist = 0
            for i in range(tileSize):
                for j in range(tileSize):
                    new_idx = patch_pos_x + j + int(local_flow[0]) + search_shift_x
                    new_idy = patch_pos_y + i + int(local_flow[1]) + search_shift_y
                    
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
    
@cuda.jit
def cuda_L2_local_search(referencePyramidLevel, alternatePyramidLevel,
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

    # position of the pixel in the top left corner of the patch
    patch_pos_x = tile_x * tileSize
    patch_pos_y = tile_y * tileSize
    
    # this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = patch_pos_x + j
            idy = patch_pos_y + i
            local_ref[i, j] = referencePyramidLevel[idy, idx]
        
    min_dist = +1/0 #init as infty
    min_shift_y = 0
    min_shift_x = 0
    # window search
    for search_shift_y in range(-searchRadius, searchRadius + 1):
        for search_shift_x in range(-searchRadius, searchRadius + 1):
            # computing dist
            dist = 0
            for i in range(tileSize):
                for j in range(tileSize):
                    new_idx = patch_pos_x + j + int(local_flow[0]) + search_shift_x
                    new_idy = patch_pos_y + i + int(local_flow[1]) + search_shift_y
                    
                    if (0 <= new_idx < w and
                        0 <= new_idy < h):
                        diff = local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx]
                        dist += diff*diff
                    else:
                        dist = +1/0
                    
                    
            if dist < min_dist :
                min_dist = dist
                min_shift_y = search_shift_y
                min_shift_x = search_shift_x
    
    upsampledAlignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
    upsampledAlignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y
    
