"""hdrplus finishing functions.
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

# imports
import os
import numpy as np
import cv2
import time
import math
import rawpy
# package-specific imports
# from 'package.algorithm'
from .genericUtils import getTime
from .imageUtils import convert8bit_
# from 'package'
from ..visualization.vis import addMotionField
from numba import guvectorize, vectorize, float32, float64, uint16, uint8, void


@vectorize([float64(float64, float64, float64, float64, float64), float32(float32, float32, float32, float32, float32)], target='parallel')
def fGammaCompress_(x, threshold, gainMin, gainMax, exponent):
	# Check the value against the threshold
	if x <= threshold:
		x = gainMin * x
	else:
		x = gainMax * (x**exponent) - gainMax + 1.
	# Clip
	return 0 if x < 0 else (1 if x > 1 else x)


@vectorize([uint16(uint16, float32, float32, float32, float32)], target='parallel')
def uGammaCompress_(x, threshold, gainMin, gainMax, exponent):
	# Normalize the image
	x /= 65535.
	# Check the value against the threshold
	if x <= threshold:
		x = gainMin * x
	else:
		x = gainMax * (x**exponent) - gainMax + 1.
	# Clip
	x *= 65535
	x = 0 if x < 0 else (65535 if x > 65535 else x)
	# Round the value to uint16
	return uint16(x)


@vectorize([float64(float64, float64, float64, float64, float64), float32(float32, float32, float32, float32, float32)], target='parallel')
def fGammaDecompress_(x, threshold, gainMin, gainMax, exponent):
	# Check the value against the threshold
	if x <= threshold:
		x = x / gainMin
	else:
		x = ((x + gainMax - 1.) / gainMax)**exponent
	# Clip
	return 0 if x < 0 else (1 if x > 1 else x)


@vectorize([uint16(uint16, float32, float32, float32, float32)], target='parallel')
def uGammaDecompress_(x, threshold, gainMin, gainMax, exponent):
	# Normalize the image
	x /= 65535.
	# Check the value against the threshold
	if x <= threshold:
		x = x / gainMin
	else:
		x = ((x + gainMax - 1.) / gainMax)**exponent
	# Clip
	x *= 65535
	x = 0 if x < 0 else (65535 if x > 65535 else x)
	# Round the value to uint16
	return uint16(x)


def gammasRGB(image, mode='compress'):
	'''sRGB transfer function'''
	if mode == 'compress':
		if np.issubdtype(image.dtype, np.unsignedinteger):
			return uGammaCompress_(image, 0.0031308, 12.92, 1.055, 1. / 2.4)
		else:
			return fGammaCompress_(image, 0.0031308, 12.92, 1.055, 1. / 2.4)
	else:
		if np.issubdtype(image.dtype, np.unsignedinteger):
			return uGammaDecompress_(image, 0.04045, 12.92, 1.055, 2.4)
		else:
			return fGammaDecompress_(image, 0.04045, 12.92, 1.055, 2.4)


def gammarec709(image, mode='compress'):
	'''rec709 transfer function'''
	if mode == 'compress':
		if np.issubdtype(image.dtype, np.unsignedinteger):
			return uGammaCompress_(image, 0.018, 4.5, 1.099, 1. / 2.2)
		else:
			return fGammaCompress_(image, 0.018, 4.5, 1.099, 1. / 2.2)
	else:
		if np.issubdtype(image.dtype, np.unsignedinteger):
			return uGammaDecompress_(image, 0.081, 4.5, 1.099, 2.2)
		else:
			return fGammaDecompress_(image, 0.081, 4.5, 1.099, 2.2)


def rgb2YUV(rgbImage):
	assert len(rgbImage.shape) == 3, 'not 3 channel image'
	assert(np.issubdtype(rgbImage.dtype, np.floating) and np.min(rgbImage) >= 0. and np.max(rgbImage) <= 1.), 'expected a float image between 0. and 1.'
	cvMat = np.array([[0.299, 0.587, 0.114],
					  [-0.14713, -0.28886, 0.436],
					  [0.615, -0.51499, -0.10001]])
	return np.dot(rgbImage, cvMat.T).clip(0., 1.)


def yuv2RGB(yuvImage):
	assert len(yuvImage.shape) == 3, 'not 3 channel image'
	assert(np.issubdtype(yuvImage.dtype, np.floating) and np.min(yuvImage) >= 0. and np.max(yuvImage) <= 1.), 'expected a float image between 0. and 1.'
	cvMat = np.array([[1, 0, 1.13983],
					  [1, -0.39465, -0.58060],
					  [1, 2.03211, 0]])
	return np.dot(yuvImage, cvMat.T).clip(0., 1.)


@vectorize([float32(float32, float32, float32), float64(float64, float64, float64)], target='parallel')
def mean_(r, g, b):
	return (r + g + b) / 3.


@vectorize([float32(float32, float32, float32, float32), float64(float64, float64, float64, float64)], target='parallel')
def meanGain_(r, g, b, k):
	# Apply gain
	rk = r * k
	gk = g * k
	bk = b * k
	# Clip the values
	rk = 0 if rk < 0 else(1 if rk > 1 else rk)
	gk = 0 if gk < 0 else(1 if gk > 1 else gk)
	bk = 0 if bk < 0 else(1 if bk > 1 else bk)
	# Average the channels
	return (rk + gk + bk) / 3.


@guvectorize(['void(float64[:, :, :], float64[:, :], float64[:, :], float64[:, :, :])'], '(h, w, c), (h, w), (h, w) -> (h, w, c)')
def applyScaling_(mergedImage, shortGray, fusedGray, result):
	for i in range(mergedImage.shape[0]):
		for j in range(mergedImage.shape[1]):
			s = 1 if shortGray[i][j] == 0 else fusedGray[i][j] / shortGray[i][j]
			for c in range(mergedImage.shape[2]):
				val = mergedImage[i][j][c] * s
				result[i][j][c] = 0 if val < 0 else (1 if val > 1 else val)


def localToneMap(burstPath, mergedImage, options):
	'''perform hdr tone mapping via exposure fusion using synthetic exposures
	(as described in Section 5.2)'''
	# Initialization
	currentTime, verbose = time.time(), options['verbose'] > 2
	burstName = os.path.basename(burstPath)

	# Work with grayscale images
	shortGray = mean_(mergedImage[:, :, 0], mergedImage[:, :, 1], mergedImage[:, :, 2])
	if verbose:
		currentTime = getTime(currentTime, ' --- Compute grayscale image')

	# Compute gain
	if options['ltmGain'] == -1:
		dsFactor = 25
		shortS = cv2.resize(shortGray, (0, 0), fx=1 / dsFactor, fy=1 / dsFactor).flatten()
		bestGain = False
		gain, compression, saturated = 0, 1., 0.
		shortSg = gammasRGB(shortS, 'compress')
		sSMean = np.mean(shortSg)
		while (compression < 1.9 and saturated < .95) or (not(bestGain) and compression < 6 and gain < 30 and saturated < 0.33):
			gain += 2
			longSg = gammasRGB(gain * shortS, 'compress').clip(0., 1.)
			lSMean = np.mean(longSg)
			compression = lSMean / sSMean
			bestGain = lSMean > (1 - sSMean) / 2  # only works if burst underexposed
			saturated = np.sum(longSg > 0.95) / np.size(longSg)
			if options['verbose'] > 4:
				print(' ----- short and long averages: ', sSMean, lSMean)
				print(' ----- compression ratio: ', compression)
				print(' ----- 95% saturation: ', saturated)
				print(' ----- Automatic selection of gain = {}'.format(gain))
	else:
		assert options['ltmGain'] > 0, 'expected a real number greater than 0'
		gain = options['ltmGain']
	if verbose:
		currentTime = getTime(currentTime, ' --- Compute gain')

	# create a synthetic long exposure
	longGray = meanGain_(mergedImage[:, :, 0], mergedImage[:, :, 1], mergedImage[:, :, 2], gain)
	if verbose:
		currentTime = getTime(currentTime, ' --- Synthetic long expo')

	# apply gamma correction to both
	shortg = gammasRGB(shortGray, 'compress')
	longg = gammasRGB(longGray, 'compress')
	if verbose:
		currentTime = getTime(currentTime, ' --- Apply Gamma correction')

	# perform tone mapping by exposure fusion in grayscale
	mergeMertens = cv2.createMergeMertens(contrast_weight=0., saturation_weight=0., exposure_weight=1.)
	if verbose:
		currentTime = getTime(currentTime, ' --- Create Mertens')
	# hack: cv2 mergeMertens expects inputs between 0 and 255
	# but the result is scaled between 0 and 1 (some values can actually be greater than 1!)
	fusedg = mergeMertens.process([255. * shortg, 255. * longg])  # .clip(0., 1.)
	if verbose:
		currentTime = getTime(currentTime, ' --- Apply Mertens')

	# undo gamma correction
	fusedGray = gammasRGB(fusedg, 'decompress')
	if verbose:
		currentTime = getTime(currentTime, ' --- Un-apply Gamma correction')

	# scale each RGB channel of the short exposure accordingly
	ltmImage = applyScaling_(mergedImage, shortGray, fusedGray)
	if verbose:
		currentTime = getTime(currentTime, ' --- Scale channels')

	return ltmImage, gain, shortg, longg, fusedg


@vectorize([float32(float32, float32), float64(float64, float64)], target='parallel')
def enhanceContrast_(x, gain):
	# Apply an S-shaped contrast enhancement curve
	x -= gain * math.sin(2 * np.pi * x)
	# Clip the result
	return 0 if x < 0 else (1 if x > 1 else x)


def enhanceContrast(image, options):
	'''perform contrast enhancement with an S-shaped function
	(as described in Section 5.2)'''
	currentTime, verbose = time.time(), options['verbose'] > 1
	# apply an S-shaped contrast enhancement curve
	assert options['gtmContrast'] >= 0 and options['gtmContrast'] <= 1, 'expected a contrast enhancement ratio between 0 and 1'
	assert(np.issubdtype(image.dtype, np.floating) and np.min(image) >= 0. and np.max(image) <= 1.), 'expected a float image between 0. and 1.'
	return enhanceContrast_(image, options['gtmContrast'])


@vectorize([float32(float32, float32), float64(float64, float64)], target='parallel')
def distL1_(x, y):
	return y - x if y > x else x - y


@vectorize([float32(float32, float32, float32, float32, float32, float32, float32, float32, float32, float32, float32, float32, float32), float64(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)], target='parallel')
def sharpenTriple_(x, b0, l0, th0, k0, b1, l1, th1, k1, b2, l2, th2, k2):
	# Compute the three sharpened values
	r0 = x if l0 < th0 else x + k0 * (x - b0)
	r1 = x if l1 < th1 else x + k1 * (x - b1)
	r2 = x if l2 < th2 else x + k2 * (x - b2)
	# Average them
	r = (r0 + r1 + r2) / 3.0
	# Clip the result
	return 0 if r < 0 else (1 if r > 1 else r)


def sharpenTriple(image, params, options):
	'''perform shapening with unsharp making
	the mask is a linear combination of convolutions of the input image
	with 3 gaussian kernels of different sizes
	(as described in Section 5.2)'''
	currentTime, verbose = time.time(), options['verbose'] > 2
	# sharpen the image using unsharp masking
	sigmas, amounts, thresholds = params['sharpenSigma'], params['sharpenAmount'], params['sharpenThreshold']

	# Compute all Gaussian blur
	blur0 = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigmas[0])
	blur1 = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigmas[1])
	blur2 = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigmas[2])
	if verbose:
		currentTime = getTime(currentTime, ' --- gaussian blur')

	# Compute all low contrast images
	low0 = distL1_(blur0, image)
	low1 = distL1_(blur1, image)
	low2 = distL1_(blur2, image)
	if verbose:
		currentTime = getTime(currentTime, ' --- low contrast')

	# Compute the triple sharpen
	sharpImage = sharpenTriple_(image, blur0, low0, thresholds[0], amounts[0], blur1, low1, thresholds[1], amounts[1], blur2, low2, thresholds[2], amounts[2])
	if verbose:
		currentTime = getTime(currentTime, ' --- sharpen')

	return sharpImage


def finish(burstPath, rawPathList, refIdx, mergedBayer, params, options):
	'''perform the finishing steps, (Section 5.2 of the IPOL article),
	and output the desired files.
	Args:
		bursPath: str, path to the folder containing the .dng files of the burst
		rawPathList: [str], list of paths to raw images in the burst folder
		refIdx: int, index of the reference image
		mergedBayer: 2D uint numpy array of a (temporally denoised) Bayer image
		params: dict containing both algorithm parameters and output choices
		options: dict containing options extracted from the script command (input/output path, mode, verbose)
	'''
	outputImage, currentTime, verbose = [], time.time(), options['verbose'] > 1
	burstName = os.path.basename(burstPath)

	# create a 16bit RGB non gamma corrected image from the merged bayer
	# (conversion to 16 bit using black and white levels, demosaicking, white balance, color correction)
	with rawpy.imread(rawPathList[refIdx]) as rawReference:
		# open the reference image file containing image metadata
		# but replace pixel values by those of the merged bayer
		rawReference.raw_image[:] = mergedBayer[:]
		processedImage = rawReference.postprocess(**params['rawpyArgs'])
	# normalize
	processedImage = processedImage / (2**16 - 1)

	if params['writeReferenceImage'] or params['writeGammaReference'] or params['writeReferenceFinal']:
		with rawpy.imread(rawPathList[refIdx]) as rawReference:
			processedRef = rawReference.postprocess(**params['rawpyArgs'])
		# normalize
		processedRef = processedRef / (2**16 - 1)

	if params['writeReferenceImage']:
		# convert to 8 bit unsigned integers
		outputImage = convert8bit_(processedRef)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed ref. image')
		outputName = os.path.join(options['outputFolder'], burstName + '_reference.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
	if params['writeGammaReference']:
		# perform sRGB gamma compression
		outputImage = gammasRGB(processedRef, 'compress')
		# convert to 8 bit unsigned integers
		outputImage = convert8bit_(outputImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed ref. image w. gamma')
		outputName = os.path.join(options['outputFolder'], burstName + '_reference_gamma.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	if params['writeMergedImage']:
		# convert to 8 bit
		outputImage = convert8bit_(processedImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed merged image')
		# save image
		outputName = os.path.join(options['outputFolder'], burstName + '_merged.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
	if params['writeGammaMerged']:
		# perform sRGB gamma compression
		outputImage = gammasRGB(processedImage, 'compress')
		outputImage = convert8bit_(outputImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed merged image w. gamma')
		outputName = os.path.join(options['outputFolder'], burstName + '_merged_gamma.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	if options['ltmGain']:
		# hdr tone mapping / local tone mapping
		processedImage, gain, shortExposure, longExposure, fusedExposure = localToneMap(burstPath, processedImage, options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Apply LTM')
		if params['writeShortExposure']:
			outputImage = convert8bit_(shortExposure)
			outputName = os.path.join(options['outputFolder'], burstName + '_short.jpg')
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
		if params['writeLongExposure']:
			outputImage = convert8bit_(longExposure)
			outputName = os.path.join(options['outputFolder'], burstName + '_longGain{}.jpg'.format(gain))
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
		if params['writeFusedExposure']:
			outputImage = convert8bit_(fusedExposure)
			outputName = os.path.join(options['outputFolder'], burstName + '_fusedGain{}.jpg'.format(gain))
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
		if params['writeLTMImage']:
			outputImage = convert8bit_(processedImage)
			outputName = os.path.join(options['outputFolder'], burstName + '_ltmGain{}.jpg'.format(gain))
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
		if params['writeLTMGamma']:
			outputImage = gammasRGB(processedImage, 'compress')
			outputImage = convert8bit_(outputImage)
			outputName = os.path.join(options['outputFolder'], burstName + '_ltmGain{}_gamma.jpg'.format(gain))
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	if options['gtmContrast']:
		# contrast enhancement / global tone mapping
		processedImage = enhanceContrast(processedImage, options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Apply GTM')

	# apply the final sRGB gamma curve
	processedImage = gammasRGB(processedImage, 'compress')
	if verbose:
		currentTime = getTime(currentTime, ' -- Apply Gamma')

	if params['writeGTMImage']:
		# convert to 8 bit
		outputImage = convert8bit_(processedImage)
		# save image
		outputName = os.path.join(options['outputFolder'], burstName + '_{}gamma.jpg'.format(('contrast{}_'.format(params['gtmContrast'])) if params['gtmContrast'] else ''))
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	# sharpen
	processedImage = sharpenTriple(processedImage, params['tuning'], options)
	if verbose:
		currentTime = getTime(currentTime, ' -- Apply sharpen')

	if params['writeFinalImage']:
		# convert to 8 bit
		outputImage = convert8bit_(processedImage)
		# save image
		outputName = os.path.join(options['outputFolder'], burstName + '_final.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	if params['writeReferenceFinal']:
		if options['ltmGain']:
			options['ltmGain'] = gain  # reuse previously computed tone mapping gain
			outputImage, _, _, _, _ = localToneMap(burstPath, processedRef, options)
		if options['gtmContrast']:
			# contrast enhancement / global tone mapping
			outputImage = enhanceContrast(outputImage, options)
		outputImage = gammasRGB(outputImage, 'compress')
		# sharpen
		outputImage = sharpenTriple(outputImage, params['tuning'], options)
		# convert to 8 bit
		outputImage = convert8bit_(outputImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Applied finishing to ref. image')
		# save image
		outputName = os.path.join(options['outputFolder'], burstName + '_reference_final.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	return
