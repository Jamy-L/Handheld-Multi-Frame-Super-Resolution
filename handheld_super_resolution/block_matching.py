# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:31:38 2022

@author: jamyl
"""
from time import time

import numpy as np
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32

from .utils import getTime, isTypeInt
from .utils_image import getTiles, getAlignedTiles, downsample, computeTilesDistanceL1_, computeDistance, subPixelMinimum

def alignBurst(ref_img, comp_imgs, params, options):
	'''Estimate motion between the reference and other images of the burst, and return a set of aligned tiles.'''
	# Initialization.
	h, w = ref_img.shape  # height and width should be identical for all images

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
	ref_img_padded = np.pad(ref_img, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric')
	comp_imgs_padded = [np.pad(im, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric') for im in comp_imgs]

	# call the HDR+ tile-based alignment function
	return alignHdrplus(ref_img_padded, comp_imgs_padded, params, options)

def alignHdrplus(referenceImage, alternateImages, params, options):
    '''Implements the coarse-to-fine alignment on 4-level gaussian pyramids
    as defined in Algorithm 1 of Section 3 of the IPOL article.
    Args:
            referenceImage / alternateImages: Bayer / grayscale images
            params: dict containing both algorithm parameters and output choices
            options: dict containing options extracted from the script command (input/output path, mode, verbose)
    '''
    # For convenience
    currentTime, verbose = time(), options['verbose'] > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = params['tuning']['factors']
    tileSizes = params['tuning']['tileSizes']
    distances = params['tuning']['distances']
    searchRadia = params['tuning']['searchRadia']
    subpixels = params['tuning']['subpixels']

    upsamplingFactors = factors[1:] + [1]
    previousTileSizes = tileSizes[1:] + [None]

    if params['mode'] == 'bayer':
        # If dealing with raw images, 2x2 bayer pixels block are averaged
        # (motion can then only be multiples of 2 pixels in original image size)
        imRef = downsample(referenceImage, kernel='bayer')
        if verbose:
            currentTime = getTime(currentTime, ' --- Ref Bayer downsampling')
        tileSize = 2 * tileSizes[0]
    else:
        imRef = referenceImage
        tileSize = tileSizes[0]

    # tiles overlap by half in each spatial dimension
    refTiles = getTiles(referenceImage, tileSize, tileSize // 2)
    alignedTiles = np.empty(
        ((len(alternateImages) + 1,) + refTiles.shape), dtype=refTiles.dtype)
    alignedTiles[0] = refTiles  # because Ref. image has no motion wrt itself
    motionVectors = np.empty(
        (len(alternateImages), refTiles.shape[0], refTiles.shape[1], 2), dtype=int)

    # construct 4-level coarse-to fine pyramid of the reference
    referencePyramid = hdrplusPyramid(imRef, factors)
    if verbose:
        currentTime = getTime(currentTime, ' --- Create ref pyramid')

    # Align each alternate image to the reference image
    for i, alternateImage in enumerate(alternateImages):

        if params['mode'] == 'bayer':
            # dowsample bayer to grayscale
            imAlt = downsample(alternateImage, kernel='bayer')
            if verbose:
                currentTime = getTime(
                    currentTime, ' --- Alt Bayer downsampling')
        else:
            imAlt = alternateImage

        # 4-level coarse-to fine pyramid of alternate image
        alternatePyramid = hdrplusPyramid(imAlt, factors)
        if verbose:
            currentTime = getTime(currentTime, ' --- Create alt pyramid')

        # succesively align from coarsest to finest level of the pyramid
        alignments = None
        for lv in range(len(referencePyramid)):
            alignments = alignOnALevel(
                referencePyramid[lv],
                alternatePyramid[lv],
                options,
                upsamplingFactors[-lv - 1],
                tileSizes[-lv - 1],
                previousTileSizes[-lv - 1],
                searchRadia[-lv - 1],
                distances[-lv - 1],
                subpixels[-lv - 1],
                alignments
            )
            
        if verbose:
            currentTime = getTime(currentTime, ' --- Align pyramid')

        # use alignment vectors to get the tiles of alternateImage matching with those of the reference image
        if params['mode'] == 'bayer':
            # estimated motion must be upsampled by a factor of 2 to go back to original image size
            alignments = upsampleAlignments(
                [], [], alignments, 2, tileSize, tileSizes[0], None)
        alignedAltTiles = getAlignedTiles(alternateImage, tileSize, alignments)
        if verbose:
            currentTime = getTime(currentTime, ' --- Get aligned tiles')
        motionVectors[i] = alignments
        alignedTiles[i + 1] = alignedAltTiles
        if verbose:
            currentTime = getTime(
                currentTime, ' -- Aligned frame {}/{} to the reference image'.format(i + 1, len(alternateImages)))

    return motionVectors, alignedTiles


def hdrplusPyramid(image, factors=[1, 2, 4, 4], kernel='gaussian'):
    '''Construct 4-level coarse-to-fine gaussian pyramid
    as described in the HDR+ paper and its supplement (Section 3.2 of the IPOL article).
    Args:
            image: input image (expected to be a grayscale image downsampled from a Bayer raw image)
            factors: [int], dowsampling factors (fine-to-coarse)
            kernel: convolution kernel to apply before downsampling (default: gaussian kernel)'''
    # Start with the finest level computed from the input
    pyramidLevels = [downsample(image, kernel, factors[0])]

    # Subsequent pyramid levels are successively created
    # with convolution by a kernel followed by downsampling
    for factor in factors[1:]:
        pyramidLevels.append(downsample(pyramidLevels[-1], kernel, factor))

    # Reverse the pyramid to get it coarse-to-fine
    return pyramidLevels[::-1]


def upsampleAlignments(referencePyramidLevel, alternatePyramidLevel, previousAlignments, upsamplingFactor, tileSize, previousTileSize, method='hdrplus'):
    '''Upsample alignements to adapt them to the next pyramid level (Section 3.2 of the IPOL article).'''
    # As resolution is multiplied, so are alignment vector values
    previousAlignments *= upsamplingFactor
    # Different resolution upsampling factors and tile sizes lead to different vector repetitions
    repeatFactor = upsamplingFactor // (tileSize // previousTileSize)
    # UpsampledAlignments.shape can be less than referencePyramidLevel.shape/tileSize
    # eg when previous alignments could not be computed over the whole image
    upsampledAlignments = previousAlignments.repeat(
        repeatFactor, 0).repeat(repeatFactor, 1)

    # If the method is not defined, no need to go further in the upsampling
    if method is None:
        return upsampledAlignments

    # For convenience
    h, w = upsampledAlignments.shape[0], upsampledAlignments.shape[1]

    #
    # HDR+ method
    #
    # Take as candidates the alignments
    # for the 3 nearest coarse-scale tiles, the nearest tile plus
    # the next-nearest neighbor tiles in each dimension

    # Pad alignments by mirroring to avoid nearest neighbor tile problems on edges
    paddedPreviousAlignments = np.pad(
        previousAlignments, pad_width=((1,), (1,), (0,)), mode='edge')

    # Create a mask of closest neighbors specifying the offsets to use to get the right indexes
    # Build the elemental tile to be repeated
    tile = np.empty((repeatFactor, repeatFactor, 2, 2), dtype=np.int)
    # upper and left coarse-scale tiles
    tile[:(repeatFactor // 2), :(repeatFactor // 2)] = [[-1, 0], [0, -1]]
    # upper and right coarse-scale tiles
    tile[:(repeatFactor // 2), (repeatFactor // 2):] = [[-1, 0], [0, 1]]
    # lower and left coarse-scale tiles
    tile[(repeatFactor // 2):, :(repeatFactor // 2)] = [[1, 0], [0, -1]]
    # lower and right coarse-scale tiles
    tile[(repeatFactor // 2):, (repeatFactor // 2):] = [[1, 0], [0, 1]]
    # Repeat the elemental tile into the full mask
    neighborsMask = np.tile(
        tile, (upsampledAlignments.shape[0] // repeatFactor, upsampledAlignments.shape[1] // repeatFactor, 1, 1))

    # Compute the indices of the neighbors using the offsets mask
    ti1 = np.repeat(np.clip(2 + np.arange(h) // repeatFactor +
                    neighborsMask[:, 0, 0, 0], 0, paddedPreviousAlignments.shape[0] - 1).reshape(h, 1), w, axis=1).reshape(h * w)
    ti2 = np.repeat(np.clip(2 + np.arange(h) // repeatFactor +
                    neighborsMask[:, 0, 1, 0], 0, paddedPreviousAlignments.shape[0] - 1).reshape(h, 1), w, axis=1).reshape(h * w)
    tj1 = np.repeat(np.clip(2 + np.arange(w) // repeatFactor +
                    neighborsMask[0, :, 0, 1], 0, paddedPreviousAlignments.shape[1] - 1).reshape(1, w), h, axis=0).reshape(h * w)
    tj2 = np.repeat(np.clip(2 + np.arange(w) // repeatFactor +
                    neighborsMask[0, :, 1, 1], 0, paddedPreviousAlignments.shape[1] - 1).reshape(1, w), h, axis=0).reshape(h * w)
    # Extract the previously estimated motion vectors associeted with those neighbors
    ppa1 = paddedPreviousAlignments[ti1, tj1].reshape((h, w, 2))
    ppa2 = paddedPreviousAlignments[ti2, tj2].reshape((h, w, 2))

    # Get all possible tiles in the reference and alternate pyramid level
    refTiles = getTiles(referencePyramidLevel, tileSize, 1)
    altTiles = getTiles(alternatePyramidLevel, tileSize, 1)

    # Compute the distance between the reference tile and the tiles that we're actually interested in
    d0 = computeTilesDistanceL1_(
        refTiles, altTiles, upsampledAlignments).reshape(h * w)
    d1 = computeTilesDistanceL1_(refTiles, altTiles, ppa1).reshape(h * w)
    d2 = computeTilesDistanceL1_(refTiles, altTiles, ppa2).reshape(h * w)

    # Get the candidate motion vectors
    candidateAlignments = np.empty((h * w, 3, 2))
    candidateAlignments[:, 0, :] = upsampledAlignments.reshape(h * w, 2)
    candidateAlignments[:, 1, :] = ppa1.reshape(h * w, 2)
    candidateAlignments[:, 2, :] = ppa2.reshape(h * w, 2)

    # Select the best candidates by keeping those that minimize the L1 distance with the reference tiles
    selectedAlignments = candidateAlignments[np.arange(
        h * w), np.argmin([d0, d1, d2], axis=0)].reshape((h, w, 2))
    if h < referencePyramidLevel.shape[0] // (tileSize // 2) - 1 or w < referencePyramidLevel.shape[1] // (tileSize // 2) - 1:
        # tiles were no alignment was computed will have an estimated motion of 0 pixels
        newAlignments = np.zeros((referencePyramidLevel.shape[0] // (
            tileSize // 2) - 1, referencePyramidLevel.shape[1] // (tileSize // 2) - 1, 2), dtype=selectedAlignments.dtype)
        newAlignments[:h, :w] = selectedAlignments
    else:
        newAlignments = selectedAlignments

    return newAlignments


def alignOnALevel(referencePyramidLevel, alternatePyramidLevel, options, upsamplingFactor=1, tileSize=16, 
                  previousTileSize=None, searchRadius=4, distance='L2', subpixel=True, previousAlignments=None):
    '''motion estimation performed at a single Gaussian pyramid level.
    Args:
            referencePyramidLevel / alternatePyramidLevel: images at the current pyramid level on which motion is estimated
            options: dict containing the level of verbosity
            upsamplingFactor: int, factor between the previous (coarser) and current pyramid level
            tileSize: current tile size
            previousTileSize: None / int, tile size used in the previous (coarser) level (if any)
            searchRadius: int, pilots the size of the square search area around the initial guess
            distance: str ('L1'/'L2'), type of norm used to compute distances between ref and alternate tiles
            subpixel: bool, whether to compute sub-pixel alignment from the pixel-level result
            previousAlignments: None / 2d array of 2d arrays, can be used to update the initial guess
    '''
    # For convenience
    verbose, currentTime = options['verbose'] > 3, time()

    # Distances shall be computed over float32 values.
    # By casting the input images here, this save the casting of much more data afterwards
    if isTypeInt(referencePyramidLevel):
        referencePyramidLevel = referencePyramidLevel.astype(np.float32)
    if isTypeInt(alternatePyramidLevel):
        alternatePyramidLevel = alternatePyramidLevel.astype(np.float32)

    # Extract the tiles of the reference image overlapped by half in each spatial dimension
    refTiles = getTiles(referencePyramidLevel, tileSize, steps=tileSize // 2)
    if verbose:
        currentTime = getTime(currentTime, ' ---- Divide in tiles')

    # Upsample the previous alignements for initialization
    if previousAlignments is None:
        # no initial offset, search windows centered around reference tiles
        upsampledAlignments = np.zeros(
            (refTiles.shape[0], refTiles.shape[1], 2), dtype=np.float32)
    else:
        # use the upsampled previous alignments as initial guesses
        upsampledAlignments = upsampleAlignments(
            referencePyramidLevel,
            alternatePyramidLevel,
            previousAlignments,
            upsamplingFactor,
            tileSize,
            previousTileSize
        )
    if verbose:
        currentTime = getTime(currentTime, ' ---- Upsample alignments')

    # For convenience
    h, w, sR = upsampledAlignments.shape[0], upsampledAlignments.shape[1], 2 * \
        searchRadius + 1

    # the initial offsets / alignment guesses [u0, v0] correspond to the upsampled previous alignments
    u0 = np.round(upsampledAlignments[:, :, 0]).astype(np.int)
    v0 = np.round(upsampledAlignments[:, :, 1]).astype(np.int)
    if verbose:
        currentTime = getTime(currentTime, ' ---- Compute indices')

    # Get all possible square search areas in the alternate image. Pad the original image with infinite values
    # each area has a side of length (tileSize + 2 * searchRadius)
    searchAreas = getTiles(np.pad(alternatePyramidLevel, searchRadius, mode='constant',
                           constant_values=2**16 - 1), window=tileSize + 2 * searchRadius)
    # Only keep those corresponding to the area around the reference tile location + [u0, v0]
    indI = np.clip(((np.repeat((np.arange(h) * tileSize // 2).reshape(h, 1), w,
                   axis=1)).reshape(h, w) + u0).reshape(h * w), 0, searchAreas.shape[0] - 1)
    indJ = np.clip(((np.repeat((np.arange(w) * tileSize // 2).reshape(1, w), h,
                   axis=0)).reshape(h, w) + v0).reshape(h * w), 0, searchAreas.shape[1] - 1)
    searchAreas = searchAreas[indI, indJ].reshape(
        (h, w, tileSize + 2 * searchRadius, tileSize + 2 * searchRadius))
    if verbose:
        currentTime = getTime(currentTime, ' ---- Extract searching areas')

    # Compute the distances between the reference tiles and each tile within the corresponding search areas
    assert(h <= refTiles.shape[0] and w <= refTiles.shape[1])
    distances = computeDistance(refTiles[:h, :w], searchAreas, distance)
    if verbose:
        currentTime = getTime(currentTime, ' ---- Compute distances')

    # Get the indexes (within the search area) of alternate tiles of minimum distance wrt the reference
    minIdx = np.argmin(distances, axis=1).astype(
        np.uint16)  # distance minimum indices are flattened

    levelOffsets = np.zeros(
        (upsampledAlignments.shape[0] * upsampledAlignments.shape[1], 2), dtype=np.float32)
    levelOffsets[..., 0], levelOffsets[..., 1] = np.unravel_index(
        minIdx, (sR, sR))  # get 2d indexes from flattened indexes
    if subpixel:
        # Initialization
        subpixOffsets = np.zeros_like(levelOffsets)
        # only work on distance minimums that actually feature a 3*3 = 9 pixel neighborhood
        validMinIdx = np.logical_and(
            minIdx < distances.shape[1] - 4, minIdx >= 4)
        # Find the optimal offset only for valid positions
        subpixOffsets[validMinIdx] = subPixelMinimum(
            distances[validMinIdx], minIdx[validMinIdx])
        # Update the offsets
        levelOffsets += subpixOffsets

    levelOffsets = levelOffsets.reshape((h, w, 2))
    # final values: initial guess [u0, v0] + current motion estimation that is actually between - and + searchRadius
    levelOffsets[..., 0] = u0 + levelOffsets[..., 0] - searchRadius
    levelOffsets[..., 1] = v0 + levelOffsets[..., 1] - searchRadius

    if verbose:
        currentTime = getTime(currentTime, ' ---- Deduce positions')

    return levelOffsets

