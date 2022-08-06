# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

@author: jamyl
"""
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.genericUtils import getTime

import cv2
import numpy as np
from time import time
import cupy as cp


def lucas_kanade_optical_flow(ref_img, comp_img, pre_alignment, options, params):
    """
    Computes the displacement based on a naive implementation of
    Lucas-Kanade optical flow (https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf)
    (The first method). This method is iterated multiple time, given by :
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

    Returns
    -------
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        alignment vector for each tile of each image

    """

    current_time, verbose = time(), options['verbose'] > 2

    tile_size = params['tuning']['tileSizes']
    n_iter = params['tuning']['kanadeIter']

    if verbose:
        current_time = time()
        print(" - Estimating Lucas-Kanade's optical flow")

    # Estimating gradients with cv2 sobel filters
    # Data format is very important. default 8uint is bad, because we use negative
    # Values, and because may go up to a value of 3080. We need signed int16
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


def lucas_kanade_optical_flow_iteration(ref_img_tiles, gradx, grady, comp_img, alignment, options, params, iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img_tiles : Array [n_tiles_y, n_tiles_x, tile_size, tile_size]
        Tiles composing the ref image (0.5 overlapping tiles)
    gradx : Array [n_tiles_y, n_tiles_x, tile_size, tile_size]
        Horizontal gradient of the ref tiles
    grady : Array [n_tiles_y, n_tiles_x, tile_size, tile_size]
        Vertical gradient of the ref tiles
    comp_img : Array[n_images, imsize_y, imsize_x]
        The images to rearrange and compare to the reference
    alignment : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The inial alignment of the tile 
    options : Dict
        Options to pass
    params : Dict
        parameters
    iter_index : int
        The iteration index (for printing evolution when verbose >2)

    Returns
    -------
    new_alignment
        Array[n_images, n_tiles_y, n_tiles_x, 2]
            The adjusted tile alignment

    """
    verbose = options['verbose']
    tile_size = params['tuning']['tileSizes']
    n_images, n_patch_y, n_patch_x, _ \
        = alignment.shape
    print(" -- Lucas-Kanade iteration {}".format(iter_index))
    # TODO this is dirty, memory usage may be doubled. We may directly call
    # cp arrays instead
    cgradx, cgrady = cp.array(gradx), cp.array(grady)
    cref_img_tiles = cp.array(ref_img_tiles)

    # aligning while considering the previous estimation
    aligned_comp_tiles = cp.empty(
        (n_images, n_patch_y, n_patch_x, tile_size, tile_size))

    # this is suboptimal but getAlignedTiles is only defined 2d arrays
    for i in range(n_images):
        aligned_comp_tiles[i] = cp.array(
            getAlignedTiles(comp_img[i], tile_size, alignment[i]))

    current_time = time()

    # TODO The next step is the bottleneck. Maybe Cupy sum and norm are not optimal
    # here, we may directly write cuda kernels

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

    # TODO The systems are 2x2. linalg.solve may be suboptimal, compared to simply
    # injecting the hand-calculated theorical solution

    # solving systems
    solution = np.linalg.solve(ATA, ATB)
    if verbose:
        current_time = getTime(
            current_time, ' \t- System solved')
    return cp.asnumpy(solution.transpose(3, 0, 1, 2))
