# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

@author: jamyl
"""
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda
from time import time
import cupy as cp
from scipy.interpolate import interp2d


def lucas_kanade_optical_flow(ref_img, comp_img, pre_alignment, options, params):
    """
    Computes the displacement based on a naive implementation of
    Lucas-Kanade optical flow (https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf)
    (The system solving method, not the iterative)

    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    pre_aligned_tiles : Array[n_images, n_tiles_y,
        n_tiles_x, tile_size, tile_size]
        The rearrangement of tile obtained after using the coarse to fine pyramidal method
    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    displacement : Array[n_images, n_tiles_y, n_tiles_x, 2]
        displacement vector for each tile of each image

    """
    # For convenience
    current_time, verbose = time(), options['verbose'] > 2

    tile_size = params['tuning']['tileSizes']
    n_iter = params['tuning']['kanade_iter']

    if verbose:
        current_time = time()
        print(" - Estimating Lucas-Kanade's optical flow")

    # Estimating gradients with cv2 sobel filters
    # Data format is very important. default 8uint is bad, because we use negative
    # Values, and because may go up to a value of 3080. We need int16
    gradx = cv2.Sobel(ref_img, cv2.CV_16S, dx=1, dy=0)
    grady = cv2.Sobel(ref_img, cv2.CV_16S, dx=0, dy=1)

    n_images, n_patch_y, n_patch_x, _ \
        = pre_alignment.shape

    # dividing into tiles, solving systems will be easier to implement
    # this way
    gradx = getTiles(gradx, tile_size, tile_size // 2)
    grady = getTiles(grady, tile_size, tile_size // 2)
    ref_img_tiles = getTiles(ref_img, tile_size, tile_size // 2)

    if verbose:
        current_time = getTime(
            current_time, ' -- Gradients estimated')

    alignment = np.array(pre_alignment, dtype=np.float64)
    for iter_index in range(n_iter):
        alignment -= lucas_kanade_optical_flow_iteration(
            ref_img_tiles, gradx, grady, comp_img, alignment, options, params, iter_index)
    return alignment


def lucas_kanade_optical_flow_iteration(ref_img_tiles, gradx, grady, comp_img, alignment, options, param, iter_index):
    verbose = options['verbose']
    tile_size = params['tuning']['tileSizes']
    n_images, n_patch_y, n_patch_x, _ \
        = pre_alignment.shape
    print(" -- Lucas-Kanade iteration {}".format(iter_index))

    # converting to cupy for gpu usage
    cgradx, cgrady = cp.array(gradx), cp.array(grady)
    cref_img_tiles = cp.array(ref_img_tiles)

    # aligning while considering the previous estimation
    aligned_comp_tiles = cp.zeros(
        (n_images, n_patch_y, n_patch_x, tile_size, tile_size))

    for i in range(n_images):
        aligned_comp_tiles[i] = cp.array(
            getAlignedTiles(comp_img[i], tile_size, alignment[i]))

    current_time = time()

    # estimating systems with adapted shape
    crossed = cp.sum(cgradx*cgrady, axis=(-2, -1))
    ATA = cp.array([[cp.linalg.norm(cgradx, axis=(-2, -1))**2, crossed],
                    [crossed, cp.linalg.norm(cgrady, axis=(-2, -1))**2]]).transpose((2, 3, 0, 1))
    ATB = cp.array([[cp.sum((aligned_comp_tiles-cref_img_tiles)*cgradx, axis=(-2, -1))],
                    [cp.sum((aligned_comp_tiles-cref_img_tiles)*cgrady, axis=(-2, -1))]])
    ATB = ATB[:, 0, :, :, :].transpose((2, 3, 0, 1))

    if verbose:
        current_time = getTime(
            current_time, ' \t- System generated')

    # solving systems
    solution = np.linalg.solve(ATA, ATB)
    if verbose:
        current_time = getTime(
            current_time, ' \t- System solved')
    return cp.asnumpy(solution.transpose(3, 0, 1, 2))


ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/payload_N000.dng'
                       ).raw_image.copy()


pre_aligned_tiles = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/33TJ_20150606_224837_294_aligned_tiles.npy')
comp_images = rawpy.imread(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]

for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)
pre_alignment = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')


params = {'tuning': {'tileSizes': 32}}
params['tuning']['kanade_iter'] = 3

final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_images, pre_alignment, {"verbose": 3}, params)

# %%

N_IMAGES, N_PATCH_Y, N_PATCH_X, _ \
    = pre_alignment.shape
TILE_SIZE = 32

comp_tiles = np.zeros((N_IMAGES, N_PATCH_Y, N_PATCH_X, TILE_SIZE, TILE_SIZE))
for i in range(N_IMAGES):
    comp_tiles[i] = getAlignedTiles(
        comp_images[i], TILE_SIZE, final_alignments[i])


# %%

native_im_size = ref_img.shape

SCALE = 2

final_frame = np.zeros(
    (SCALE*native_im_size[0], SCALE*native_im_size[1], 3))  # * float('Nan')
acc = np.zeros((final_frame.shape))

comp_tiles_dummy = np.zeros(comp_tiles.shape)
# %%

# https://stackoverflow.com/questions/56008378/calling-other-functions-from-within-a-cuda-jit-numba-function


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


@cuda.jit(device=True)
def compute_rgb(neighborhood_idy, neighborhood_idx,
                subtile_center_idy, subtile_center_idx,
                neighborhood, rgb, acc):

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 0 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 0):  # Bayer 00
        if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # r
            rgb[0] += neighborhood[neighborhood_idy, neighborhood_idx]
            acc[0] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # g top
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # g bottom
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx - 1] != np.inf:  # g left
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx - 1]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx + 1] != np.inf:  # g right
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx + 1]
            acc[1] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # b top left
            rgb[2] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[2] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # b top right
            rgb[2] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[2] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # b bottom left
            rgb[2] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[2] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # b bottom right
            rgb[2] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[2] += 1

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 0 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 1):  # Bayer 01
        if neighborhood[neighborhood_idy, neighborhood_idx-1] != np.inf:  # r left
            rgb[0] += neighborhood[neighborhood_idy, neighborhood_idx-1]
            acc[0] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx+1] != np.inf:  # r right
            rgb[0] += neighborhood[neighborhood_idy, neighborhood_idx+1]
            acc[0] += 1

        if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # g
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # g top left
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # g top right
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # g bottom left
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # g bottom right
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[1] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # b top
            rgb[2] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[2] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # b bottom
            rgb[2] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[2] += 1

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 1 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 0):  # Bayer 10
        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # r top
            rgb[0] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[0] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # r bottom
            rgb[0] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[0] += 1

        if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # g
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # g top left
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # g top right
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # g bottom left
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # g bottom right
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[1] += 1

        if neighborhood[neighborhood_idy, neighborhood_idx-1] != np.inf:  # b left
            rgb[2] += neighborhood[neighborhood_idy, neighborhood_idx-1]
            acc[2] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx+1] != np.inf:  # b right
            rgb[2] += neighborhood[neighborhood_idy, neighborhood_idx+1]
            acc[2] += 1

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 1 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 1):  # Bayer 11
        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # r top left
            rgb[0] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[0] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # r top right
            rgb[0] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[0] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # r bottom left
            rgb[0] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[0] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # r bottom right
            rgb[0] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[0] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # g top
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # g bottom
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx - 1] != np.inf:  # g left
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx - 1]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx + 1] != np.inf:  # g right
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx + 1]
            acc[1] += 1

    if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # b
        rgb[2] += neighborhood[neighborhood_idy, neighborhood_idx]
        acc[2] += 1


@cuda.jit
def accumulate(comp_tiles, final_alignments, final_frame, accumulator, comp_tiles_dummy):

    image_index, tile_idx, tile_idy = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z,
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # calculating the coordinates of the center of the subtiles in the
    # referential of the tile
    subtile_center_idx = 3*tx+1
    subtile_center_idy = 3*ty+1

    if not (0 <= image_index < N_IMAGES and 0 <= tile_idx < N_PATCH_X and
            0 <= tile_idy < N_PATCH_Y):
        return

    # all threads are executing on the same tile. Let's store it in sharred memory
    tile = cuda.shared.array(
        shape=(TILE_SIZE, TILE_SIZE), dtype=float64)
    alignment = cuda.shared.array(
        shape=(2), dtype=float64)

    alignment[0] = final_alignments[image_index, tile_idy, tile_idx, 0]
    alignment[1] = final_alignments[image_index, tile_idy, tile_idx, 1]

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

    # extracting the large neighborhood (5x5)
    neighborhood = cuda.local.array(shape=(5, 5), dtype=float64)

    rgb = cuda.local.array(shape=(3), dtype=float64)
    acc = cuda.local.array(shape=(3), dtype=float64)

    extract_neighborhood(subtile_center_idy,
                         subtile_center_idx, tile, neighborhood)

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

            scaled_coordinate_y = int((subtile_center_idy + (neighborhood_idy - 2)
                                       + TILE_SIZE//2*tile_idy + alignment[1])*SCALE)
            scaled_coordinate_x = int((subtile_center_idx + (neighborhood_idx - 2)
                                       + TILE_SIZE//2*tile_idx + alignment[0])*SCALE)

            if ((0 <= scaled_coordinate_y < final_frame.shape[0]) and
                (0 <= scaled_coordinate_x < final_frame.shape[1]) and
                (0 <= subtile_center_idy + (neighborhood_idy - 2) < TILE_SIZE) and
                    (0 <= subtile_center_idx + (neighborhood_idx - 2) < TILE_SIZE)):

                cuda.atomic.add(final_frame,
                                (scaled_coordinate_y, scaled_coordinate_x, 0),
                                rgb[0])
                cuda.atomic.add(final_frame,
                                (scaled_coordinate_y, scaled_coordinate_x, 1),
                                rgb[1])
                cuda.atomic.add(final_frame,
                                (scaled_coordinate_y, scaled_coordinate_x, 2),
                                rgb[2])

                cuda.atomic.add(accumulator,
                                (scaled_coordinate_y, scaled_coordinate_x, 0),
                                acc[0])
                cuda.atomic.add(accumulator, (scaled_coordinate_y, scaled_coordinate_x, 1),
                                acc[1])
                cuda.atomic.add(accumulator, (scaled_coordinate_y, scaled_coordinate_x, 2),
                                acc[2])


# moving arrays to GPU
cuda_comp_tiles = cuda.to_device(comp_tiles)
cuda_comp_tiles_dummy = cuda.to_device(comp_tiles_dummy)
cuda_final_alignments = cuda.to_device(final_alignments)
cuda_final_frame = cuda.to_device(final_frame)
cuda_accumulator = cuda.to_device(acc)


# specifying the block size
threadsperblock = (int(np.ceil(TILE_SIZE/3)), int(np.ceil(TILE_SIZE/3)))
blockspergrid_x = N_PATCH_X
blockspergrid_y = N_PATCH_Y
blockspergrid = (N_IMAGES, N_PATCH_X, N_PATCH_Y)

accumulate[blockspergrid, threadsperblock](
    cuda_comp_tiles, cuda_final_alignments, cuda_final_frame, cuda_accumulator,
    cuda_comp_tiles_dummy)

final_frame = cuda_final_frame.copy_to_host()
accumulator = cuda_accumulator.copy_to_host()
comp_tiles_dummy = cuda_comp_tiles_dummy.copy_to_host()
accumulator[accumulator == 0] = 1
# %%
a = final_frame/accumulator
