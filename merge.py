# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_rgb

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda
from time import time
import cupy as cp
from scipy.interpolate import interp2d


def merge(ref_img, comp_tiles, alignments, options, params):
    """
    Merges all the images, based on the alignments previously estimated.
    The size of the merge_result is adjustable with params['scale']


    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    comp_tiles : Array [n_images, n_tiles_y, n_tiles_x, tile_size, tile_size]
        Tiles composing the compared image (0.5 overlapping tiles)
    alignments : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment
    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    merge_result : Array[scale * imsize_y, scale * imsize_x, 3]
        merged images

    """
    # TODO Non int scale is broken
    VERBOSE = options['verbose']
    SCALE = params['scale']
    TILE_SIZE = params['tuning']['tileSizes']
    N_IMAGES, N_TILES_Y, N_TILES_X, _ \
        = alignments.shape

    assert TILE_SIZE == comp_tiles.shape[3] == comp_tiles.shape[4], (
        "Mismatch between announced tile size ({}) and comp_tiles tile"
        " size ({} by {})". format(TILE_SIZE,
                                   comp_tiles.shape[3],
                                   comp_tiles.shape[4]))

    if VERBOSE > 2:
        print('Beginning mergin process')
        current_time = time()

    native_im_size = ref_img.shape
    # TODO we may rework the init to optimize memory usage and bandwidth usage
    # betweeen host and device
    merge_result = np.zeros(
        (SCALE*native_im_size[0], SCALE*native_im_size[1], 3))
    accumulator = np.zeros((merge_result.shape))

    # moving arrays to GPU
    cuda_comp_tiles = cuda.to_device(comp_tiles)
    cuda_alignments = cuda.to_device(alignments)
    cuda_merge_result = cuda.to_device(merge_result)
    cuda_accumulator = cuda.to_device(accumulator)

    # specifying the block size
    # TODO 1 bloc = 1 tile so far. Is that the best choice ??
    threadsperblock = (int(np.ceil(TILE_SIZE/3)), int(np.ceil(TILE_SIZE/3)))
    blockspergrid = (N_IMAGES, N_TILES_X, N_TILES_Y)

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data transfered to GPU')

    @cuda.jit(device=True)
    def extract_neighborhood(subtile_center_idy, subtile_center_idx, tile, neighborhood):
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (0 <= subtile_center_idy + i < TILE_SIZE and
                        0 <= subtile_center_idx + j < TILE_SIZE):

                    neighborhood[i+2, j+2] = tile[subtile_center_idy + i,
                                                  subtile_center_idx + j]
                else:
                    # TODO Nan would be better but I don't know how to create Nan with cuda
                    neighborhood[i+2, j+2] = np.inf

    @cuda.jit
    def accumulate(comp_tiles, alignments, merge_result, accumulator):
        """
        Cuda kernel, each thread represents a 3x3 neighborhood within a
        specific tile of a specific image. These neighborhoods dont overlap
        and cover the entirety of comp_tiles.



        Parameters
        ----------
        comp_tiles : Array [n_images, n_tiles_y, n_tiles_x, tile_size, tile_size]
            Tiles composing the compared image (0.5 overlapping tiles)
        alignments :  Array[n_images, n_tiles_y, n_tiles_x, 2]
            The final estimation of the tiles' alignment
        merge_result : Array[scale * imsize_y, scale * imsize_x, 3]
            The numerator of pixel values
        accumulator : Array[scale * imsize_y, scale * imsize_x, 3]
            The denominator of pixel values

        Returns
        -------
        None.

        """

        image_index, tile_idx, tile_idy = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z,
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # calculating the coordinates of the center of the subtiles in the
        # referential of the tile
        subtile_center_idx = 3*tx+1
        subtile_center_idy = 3*ty+1

        # TODO Apparently conditonal return and cuda.syncthreads() dont mix well,
        # The code as such works, but we may investigate
        if not (0 <= image_index < N_IMAGES and 0 <= tile_idx < N_TILES_X and
                0 <= tile_idy < N_TILES_Y):
            return

        # TODO is sharred memory useful or not here ?
        # all threads are executing on the same tile. Let's store it in sharred memory
        tile = cuda.shared.array(
            shape=(TILE_SIZE, TILE_SIZE), dtype=float64)
        alignment = cuda.shared.array(
            shape=(2), dtype=float64)

        alignment[0] = alignments[image_index, tile_idy, tile_idx, 0]
        alignment[1] = alignments[image_index, tile_idy, tile_idx, 1]

        for i in range(-1, 2):
            for j in range(-1, 2):
                if (0 <= subtile_center_idy+i < TILE_SIZE and
                        0 <= subtile_center_idx+j < TILE_SIZE):

                    tile[subtile_center_idy+i, subtile_center_idx + j] =\
                        comp_tiles[image_index, tile_idy, tile_idx,
                                   subtile_center_idy+i, subtile_center_idx + j]

        # syncing when the entire thread pool is done
        cuda.syncthreads()

        # TODO calculating the kernel and the confidence once for each thread

        # TODO This is very memory expensive. Maybe it can be avoided.
        # extracting the large neighborhood (5x5)
        neighborhood = cuda.local.array(shape=(5, 5), dtype=float64)

        extract_neighborhood(subtile_center_idy,
                             subtile_center_idx, tile, neighborhood)

        rgb = cuda.local.array(shape=(3), dtype=float64)
        acc = cuda.local.array(shape=(3), dtype=float64)

        for neighborhood_idy in range(1, 4):
            for neighborhood_idx in range(1, 4):
                rgb[0] = 0.
                rgb[1] = 0.
                rgb[2] = 0.
                acc[0] = 0
                acc[1] = 0
                acc[2] = 0
                compute_rgb(neighborhood_idy, neighborhood_idx,
                            subtile_center_idy, subtile_center_idx,
                            neighborhood, rgb, acc)

                # corresponding coordinates in the merge_result referential
                scaled_coordinate_y = int((subtile_center_idy + (neighborhood_idy - 2)
                                           + TILE_SIZE//2*tile_idy + alignment[1])*SCALE)
                scaled_coordinate_x = int((subtile_center_idx + (neighborhood_idx - 2)
                                           + TILE_SIZE//2*tile_idx + alignment[0])*SCALE)

                # 2 condition:
                #   ->  After dilation and translation, the pixel
                #       coordinates must arrive in the merge_result.
                #   ->  Tile_size is not divible by 3 (but by 2), so
                #       some subtiles are slightly out of tile
                if ((0 <= scaled_coordinate_y < merge_result.shape[0]) and
                    (0 <= scaled_coordinate_x < merge_result.shape[1]) and
                    (0 <= subtile_center_idy + (neighborhood_idy - 2) < TILE_SIZE) and
                        (0 <= subtile_center_idx + (neighborhood_idx - 2) < TILE_SIZE)):

                    # Atomic operations are madatory ! (no +=)
                    cuda.atomic.add(merge_result,
                                    (scaled_coordinate_y, scaled_coordinate_x, 0),
                                    rgb[0])
                    cuda.atomic.add(merge_result,
                                    (scaled_coordinate_y, scaled_coordinate_x, 1),
                                    rgb[1])
                    cuda.atomic.add(merge_result,
                                    (scaled_coordinate_y, scaled_coordinate_x, 2),
                                    rgb[2])

                    cuda.atomic.add(accumulator,
                                    (scaled_coordinate_y, scaled_coordinate_x, 0),
                                    acc[0])
                    cuda.atomic.add(accumulator, (scaled_coordinate_y, scaled_coordinate_x, 1),
                                    acc[1])
                    cuda.atomic.add(accumulator, (scaled_coordinate_y, scaled_coordinate_x, 2),
                                    acc[2])

    accumulate[blockspergrid, threadsperblock](
        cuda_comp_tiles, cuda_alignments, cuda_merge_result, cuda_accumulator)

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data merged on GPU side')

    # TODO This can probably be modified to save memory, but so far it's good
    # for debuging
    merge_result = cuda_merge_result.copy_to_host()
    accumulator = cuda_accumulator.copy_to_host()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data returned from GPU')

    # To avoid division by zeros
    # accumulator[accumulator == 0] = 1

    # TODO 1023 is the max value in the example. Maybe we have to extract
    # metadata to have a more general framework
    # return merge_result/(1023*accumulator)
    return merge_result, accumulator

# %% test code, to remove in the final version


ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/payload_N000.dng'
                       ).raw_image.copy()


comp_images = rawpy.imread(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)

pre_alignment = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')
n_images, n_patch_y, n_patch_x, _ = pre_alignment.shape
tile_size = 32
native_im_size = ref_img.shape

params = {'tuning': {'tileSizes': 32}, 'scale': 2}
params['tuning']['kanadeIter'] = 3


t1 = time()
final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_images, pre_alignment, {"verbose": 3}, params)

comp_tiles = np.zeros((n_images, n_patch_y, n_patch_x, tile_size, tile_size))
for i in range(n_images):
    comp_tiles[i] = getAlignedTiles(
        comp_images[i], tile_size, final_alignments[i])


num, den = merge(ref_img, comp_tiles, final_alignments, {"verbose": 3}, params)
print('\nTotal ellapsed time : ', time() - t1)
