
""" Utility functions that work on images / arrays of images.
Copyright (c) 2021 Antoine Monod

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

This file implements an algorithm possibly linked to the patent US9077913B2.
This file is made available for the exclusive aim of serving as scientific tool
to verify the soundness and completeness of the algorithm description.
Compilation, execution and redistribution of this file may violate patents rights in certain countries.
The situation being different for every country and changing over time,
it is your responsibility to determine which patent rights restrictions apply to you
before you compile, use, modify, or redistribute this file.
A patent lawyer is qualified to make this determination.
If and only if they don't conflict with any patent terms,
you can benefit from the following license terms attached to this file.
"""

import math

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from numba import vectorize, guvectorize, uint8, uint16, float32, float64

from .utils import getSigned, isTypeInt
from .fft import fft2, ifft2


@vectorize([uint8(float32), uint8(float64)], target='parallel')
def convert8bit_(x):
    return 0 if x <= 0 else (255 if x >= 1 else uint8(x * 255 + 0.5))

@vectorize([uint8(float32), uint8(float64)], target='parallel')
def convert16bit_(x):
    return 0 if x <= 0 else ((2**16 - 1) if x >= 1 else uint16(x * (2**16 - 1) + 0.5))

@vectorize([uint16(uint16, uint16, uint16, uint16)], target='parallel')
def umean4_(a, b, c, d):
    return np.right_shift(a + b + c + d + 2, 2)


@vectorize([float32(float32, float32, float32, float32), float64(float64, float64, float64, float64)], target='parallel')
def fmean4_(a, b, c, d):
    return (a + b + c + d) * 0.25


def downsample(image, kernel='gaussian', factor=2):
    '''Apply a convolution by a kernel if required, then downsample an image.
    Args:
    	image: the input image (WARNING: single channel only!)
    	kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
    	factor: downsampling factor
    '''
    # Special case
    if factor == 1:
    	return image

    # Filter the image before downsampling it
    if kernel is None:
    	filteredImage = image
    elif kernel == 'gaussian':
    	# gaussian kernel std is proportional to downsampling factor
    	filteredImage = gaussian_filter(image, sigma=factor * 0.5, order=0, output=None, mode='reflect')
    elif kernel == 'bayer':
    	# Bayer means that a simple 2x2 aggregation is required
    	if isTypeInt(image):
    	    return umean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
    	else:
    	    return fmean4_(image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2])
    else:
    	# filter by convoluting with the input kernel
    	filteredImage = signal.convolve2d(image, kernel, boundary='symm', mode='valid')

    # Shape of the downsampled image
    h2, w2 = np.floor(np.array(filteredImage.shape) / float(factor)).astype(np.int)

    # Extract the pixels
    if isTypeInt(image):
    	return np.rint(filteredImage[:h2 * factor:factor, :w2 * factor:factor]).astype(image.dtype)
    else:
    	return filteredImage[:h2 * factor:factor, :w2 * factor:factor]


def getAlignedTiles(image, tileSize, motionVectors):
    '''Replace tiles within an image based on estimated motion between
    said image and a reference image image.'''
 
    # For convenience
    # total number of image tiles overlapped by half in each spatial dimension: h*w
    h, w = image.shape[0] // (tileSize // 2) - 1, image.shape[1] // (tileSize // 2) - 1
    # total number of tiles that have estimated motion : hm*wm (<= h*w)
    hm, wm, _ = motionVectors.shape
    # Extract all the possible tiles of the image
    imageTiles = getTiles(image, tileSize)
    # Get the indices of the beginning of each tiles when moved
    indIm = np.round((np.repeat((np.arange(hm) * tileSize // 2).reshape(hm, 1), wm, axis=1) + motionVectors[:, :, 0])).astype(np.int).clip(0, image.shape[0] - tileSize)
    indJm = np.round((np.repeat((np.arange(wm) * tileSize // 2).reshape(1, wm), hm, axis=0) + motionVectors[:, :, 1])).astype(np.int).clip(0, image.shape[1] - tileSize)
    indI = np.repeat((np.arange(h) * tileSize // 2).reshape(h, 1), w, axis=1)
    indJ = np.repeat((np.arange(w) * tileSize // 2).reshape(1, w), h, axis=0)
    indI[:indIm.shape[0], :indIm.shape[1]] = indIm  # tiles that have no motion vectors attached will remain identical
    indJ[:indJm.shape[0], :indJm.shape[1]] = indJm

    # Only keep the tiles we're interested in (step = tileSize // 2)
    alignedTiles = imageTiles[indI.reshape(h * w), indJ.reshape(h * w)].reshape((h, w, tileSize, tileSize))

    return alignedTiles


@guvectorize(['void(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :], float32[:, :])'], '(m, n, t, t), (m, n, t, t), (h, w, d) -> (h, w)')
def computeTilesDistanceL1_(refTiles, altTiles, offsets, res):
    '''Compute the L1 distance between refTiles and extracted altTiles from the offsets'''
    # Dimension
    m, n, ts, _ = refTiles.shape
    h, w, _ = offsets.shape

    # Loop over the aligned tiles
    for i in range(h):
        for j in range(w):
            # Offset values
            offI = offsets[i, j, 0]
            offJ = offsets[i, j, 1]
            # Reference index
            ri = i * (ts // 2)
            rj = j * (ts // 2)
            # Deduce the position of the corresponding tiles
            di = ri + int(offI + (0.5 if offI >= 0 else -0.5))
            dj = rj + int(offJ + (0.5 if offJ >= 0 else -0.5))
            # Clip the position
            di = 0 if di < 0 else (m - 1 if di > m - 1 else di)
            dj = 0 if dj < 0 else (n - 1 if dj > n - 1 else dj)
            # Compute the distance
            dst = 0
            for p in range(ts):
                for q in range(ts):
                    dst += math.fabs(refTiles[ri, rj, p, q] - altTiles[di, dj, p, q])
            # Store the result
            res[i, j] = dst


@guvectorize(['void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])'], '(n, w, w), (n, p, p), (n, t, t) -> (n, t, t)')
def computeL1Distance_(win, ref, dum, res):
    # Dummy array dum only here to know the output size. Won't be used.
    # Get the shapes
    hw, sW, sP, sT = win.shape[0], win.shape[1], ref.shape[1], win.shape[1] - ref.shape[1] + 1
    # Loop over all the pixels of the image
    for n in range(hw):
        # Extract all the generic patches in the current searching windows
        for i in range(sT):
    	    for j in range(sT):
    	        # Distance computation
    	        sum = 0
    	        for p in range(sP):
    	            for q in range(sP):
    	                sum += math.fabs(win[n, i + p, j + q] - ref[n, p, q])
                # Store the distance
    	        res[n, i, j] = sum


# ## NOTE: old implementation.
# @guvectorize(['void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])'], '(n, w, w), (n, p, p), (n, t, t) -> (n, t, t)')
# def computeL2Distance__(win, ref, dum, res):
#     # Dummy array dum only here to know the output size. Won't be used.
#     # Get the shapes
#     hw, sW, sP, sT = win.shape[0], win.shape[1], ref.shape[1], win.shape[1] - ref.shape[1] + 1
#     # Loop over all the pixels of the image
#     for n in range(hw):
#         # Extract all the generic patches in the current searching windows
#         for i in range(sT):
#             for j in range(sT):
#                 # Distance computation
#                 sum = 0
#                 for p in range(sP):
#                     for q in range(sP):
#                         sum += (win[n, i + p, j + q] - ref[n, p, q])**2
#                 # Store the distance
#                 res[n, i, j] = sum


## Note: FFT-based L2 distance computation. It may be slow anyway because of homemade numba FFT.
@guvectorize(['void(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :])'], '(n, w, w), (n, p, p), (n, t, t) -> (n, t, t)')
def computeL2Distance_(win, ref, dum, res):
    # Dummy array dum only here to know the output size. Won't be used.
    # Get the shapes: sW = m, sP = n, sT = m-n+1
    hw, sW, sP, sT = win.shape[0], win.shape[1], ref.shape[1], win.shape[1] - ref.shape[1] + 1
    # Compute the norm of T once -> globally summed squared entries of T
    for n in range(hw):
        res[n] = np.sum(ref[n] * ref[n])
    # FFT-based box filtering of I -> slided summed squared entries of I
    # FFT and IFFT are done on the two last entries of the arrays (cf .fft.py)
    box = np.zeros((sW, sW), dtype=win.dtype)
    box[:sP, :sP] = np.ones((sP, sP), dtype=win.dtype)
    res += ifft2(fft2(box) * fft2(win * win)).real[..., sP-1:sP-1+sT, sP-1:sP-1+sT]  # (n,t,t)
    # FFT-based cross-correlation between T and I
    ref_padded = np.zeros_like(win)
    ref_padded[..., :sP, :sP] = ref  # do zero-padding to fit win's size
    res -= 2 * ifft2(fft2(ref_padded).conjugate() * fft2(win)).real[..., :sT, :sT]  # (n,t,t) -- ref must be conjugated since it is the "filter"


def computeDistance(refPatch, searchArea, distance='L2'):
    '''Return the distance between the reference patch and all the ones contained in the searching area,
    by using the L1 or L2 distance norm.'''
    # For convenience
    h, w, sP, sP2 = refPatch.shape
    hs, ws, sW, sW2 = searchArea.shape
    # Assert that inputs are consistants
    assert((sP == sP2) and (hs == h) and (ws == w) and (sW == sW2))

    # Reshape the input
    win = searchArea.reshape(h * w, sW, sW)
    ref = refPatch.reshape(h * w, sP, sP)
    # Dummy input just to pass the output dimension to numba
    dum = np.empty((h * w, sW - sP + 1, sW - sP + 1), dtype=ref.dtype)

    # Compute the distance between the reference patch and all patches in the searching area
    if distance == 'L1':
        dst = computeL1Distance_(win, ref, dum)
    elif distance == 'L2':
        dst = computeL2Distance_(win, ref, dum)
    else:
        print('Unknown distance ' + distance + '. Abort.')
        exit()

    # Reshape the output
    return dst.reshape(h * w, (sW - sP + 1)**2)


@guvectorize(['void(float32[:, :], uint16[:], float32[:], float32[:])'], '(m, n), (m) -> (m), (m)')
def subPixelMinimum_(arrDst, arrIdx, resI, resJ):
    '''To compute the sub-pixel offset to the distance minimum (Section 3.3 of the IPOL article),
    fit a bivariate quadratic function (2D polynomial)
    to the 3x3 neighborhood around the pixel-level minimum, and find the minimum of the quadratic.
    Args:
    	arrDst: array of arrays of distances
    		(rows: reference tile; columns: distances of all possible tiles within the search area)
    	arrIdx: array of corresponding indices of the distance minima
    '''
    # Filters that are used to construct matrix A and vector b
    # A and b are parameters of the quadratic approximation
    fA11 = [ 0.250, -0.50,  0.250,  0.50, -1.,  0.50,  0.250, -0.50, 0.250]
    fA22 = [ 0.250,  0.50,  0.250, -0.50, -1., -0.50,  0.250,  0.50, 0.250]
    fA12 = [ 0.250,  0.00, -0.250,  0.00,  0.,  0.00, -0.250,  0.00, 0.250]
    fb1  = [-0.125,  0.00,  0.125, -0.25,  0.,  0.25, -0.125,  0.00, 0.125]
    fb2  = [-0.125, -0.25, -0.125,  0.00,  0.,  0.00,  0.125,  0.25, 0.125]

    for m in range(len(arrIdx)):
        ind = arrIdx[m]
        
        # Construct A and b
        # by cross-correlating filters with the 3x3 neighborhoods
        A11, A12, A22, b1, b2 = 0, 0, 0, 0, 0
        for i in range(9):
            d = arrDst[m, ind - 4 + i]
            A11 += fA11[i] * d
            A12 += fA12[i] * d
            A22 += fA22[i] * d
            b1  += fb1 [i] * d
            b2  += fb2 [i] * d

        # Make sure that A is positive semi-definite
        A11 = max(0, A11)
        A22 = max(0, A22)
        if A11 * A22 - A12**2 < 0:
            # Put all off-diagonal values of A to zero
            A12 = 0

        # Compute the determinant
        det = A11 * A22 - A12**2
 
        # If null, return 0
        if det == 0:
            resI[m] = 0
            resJ[m] = 0
        else:
            # Compute the offset vector
            # it is the minimum of the quadratic mu = - A^(-1) b
            osvI = -(A11 * b2 - A12 * b1) / det
            osvJ = -(A22 * b1 - A12 * b2) / det
            # Compute the norm of this vector
            nrm = math.sqrt(osvI**2 + osvJ**2)
            # Only keep it if it is less than 1 pixel away
            resI[m] = osvI * (nrm < 1)
            resJ[m] = osvJ * (nrm < 1)


def subPixelMinimum(arrDst, arrIdx):
    '''Compute sub-pixel distance minima
    from 3x3 neighborhoods around the pixel-level minima.
    (Section 3.3 of the IPOL article).
    Args:
    	arrDst: array of arrays of distances
    		(rows: reference tile; columns: distances of all possible tiles within the search area)
    	arrIdx: array of corresponding indices of the distance minima
    '''
    # Initialization
    offsets = np.empty((len(arrIdx), 2), dtype=arrDst.dtype)
    # Compute the minimum
    offsets[:, 0], offsets[:, 1] = subPixelMinimum_(arrDst, arrIdx)
    # Return the offsets
    return offsets


def computeRMSE(image1, image2):
    '''computes the Root Mean Square Error between two images'''
    assert np.array_equal(image1.shape, image2.shape), 'images have different sizes'
    h, w = image1.shape[:2]
    c = 1
    if len(image1.shape) == 3:  # multi-channel image
        c = image1.shape[-1]
    error = getSigned(image1.reshape(h * w * c)) - getSigned(image2.reshape(h * w * c))
    return np.sqrt(np.mean(np.multiply(error, error)))


def computePSNR(image, noisyImage):
    '''computes the Peak Signal-to-Noise Ratio between a "clean" and a "noisy" image'''
    if np.array_equal(image.shape, noisyImage.shape):
        assert image.dtype == noisyImage.dtype, 'images have different data types'
        if np.issubdtype(image.dtype, np.unsignedinteger):
            maxValue = np.iinfo(image.dtype).max
        else:
            assert(np.issubdtype(image.dtype, np.floating) and np.min(image) >= 0. and np.max(image) <= 1.), 'not a float image between 0 and 1'
            maxValue = 1.
        h, w = image.shape[:2]
        c = 1
        if len(image.shape) == 3:  # multi-channel image
            c = image.shape[-1]
        error = np.abs(getSigned(image.reshape(h * w * c)) - getSigned(noisyImage.reshape(h * w * c)))
        mse = np.mean(np.multiply(error, error))
        return 10 * np.log10(maxValue**2 / mse)
    else:
        print('WARNING: images have different sizes: {}, {}. Returning None'.format(image.shape, noisyImage.shape))
        return None


## NOTE: this returns a (h,w,H,W) tile collections, each size (H,W)
def getTiles(a, window, steps=None, axis=None):
    '''
    Create a windowed view over `n`-dimensional input that uses an
    `m`-dimensional window, with `m <= n`
    
    Parameters
    -------------
    a : Array-like
    	The array to create the view on
    
    window : tuple or int
    	If int, the size of the window in `axis`, or in all dimensions if
    	`axis == None`
    
    	If tuple, the shape of the desired window.  `window.size` must be:
    		equal to `len(axis)` if `axis != None`, else
    		equal to `len(a.shape)`, or
    		1
    
    steps : tuple, int or None
    	The offset between consecutive windows in desired dimension
    	If None, offset is one in all dimensions
    	If int, the offset for all windows over `axis`
    	If tuple, the steps along each `axis`.
    		`len(steps)` must me equal to `len(axis)`
    
    axis : tuple, int or None
    	The axes over which to apply the window
    	If None, apply over all dimensions
    	if tuple or int, the dimensions over which to apply the window
    
    Returns
    -------
    
    a_view : ndarray
    	A windowed view on the input array `a`, or a generator over the windows
    
    '''
    ashp = np.array(a.shape)

    if axis is not None:
        axs = np.array(axis, ndmin=1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin=1)
    assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"
    
    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin=1)
        assert np.all(steps > 0), "Only positive steps allowed"
        assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
        stp[axs] = steps

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    return np.squeeze(np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides))



if __name__ == '__main__':
    from skimage import data, img_as_float32
    import time
    img = img_as_float32(data.astronaut())
    img = img.mean(-1)
    image1 = img[100:130, 100:145]
    image2 = np.roll(img, (2,3))[100:130, 100:145]
    win = 16
    radius = 4

    tiles1 = getTiles(image1, win +2*radius)
    tiles1 = tiles1[...,radius:-radius, radius:-radius]
    tiles2 = getTiles(image2, win +2*radius)
    print(tiles1.shape, tiles2.shape)

    # for convinience
    h, w, sP, sP2 = tiles1.shape
    hs, ws, sW, sW2 = tiles2.shape

    # Reshape the input
    win = tiles2.reshape(h * w, sW, sW)
    ref = tiles1.reshape(h * w, sP, sP)
    # Dummy input just to pass the output dimension to numba
    dum = np.empty((h * w, sW - sP + 1, sW - sP + 1), dtype=ref.dtype)

    # Compute the distance between the reference patch and all patches in the searching area
    n = 1
    # n = h*w
    dst_fft = computeL2Distance_(win[:n], ref[:n], dum[:n])  # new implementation: based on FFT
    # dst_direct = computeL2Distance_(win[:n], ref[:n], dum[:n])  # old implementation: nested for loops

    print(np.linalg.norm(dst_direct - dst_fft) / np.prod(dst_fft.shape))
