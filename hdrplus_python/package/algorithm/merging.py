"""hdrplus burst merging functions.
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
import shutil
import time
import numpy as np
import rawpy
import cv2
# package-specific imports
# from 'package.algorithm'
from .genericUtils import getTime
from .imageUtils import convert8bit_
from .finishing import gammasRGB
from numba import vectorize, float32, float64, complex64, complex128
import torch.fft


def mergeBurst(burstPath, rawPathList, images, referenceIndex, alignedTiles, padding, tags, blackLevel, whiteLevel, params, options):
	'''Merged previously aligned tiles of a burst, and return a single temporally denoised (Bayer) image.'''
	currentTime, verbose = time.time(), options['verbose'] > 2
	mergedImage = mergeHdrplus(images[referenceIndex], alignedTiles, padding, tags, blackLevel, whiteLevel, params['tuning'], options)
	burstName = os.path.basename(burstPath)

	if params['writeMergedBayer']:
		mergedFolder = os.path.join(options['outputFolder'], burstName)
		if not os.path.isdir(mergedFolder):
			os.mkdir(mergedFolder)
		# copy the reference image .dng file
		shutil.copy2(rawPathList[referenceIndex], os.path.join(mergedFolder, os.path.basename(rawPathList[referenceIndex])))
		# save merged bayer as .npy file
		np.save(os.path.join(mergedFolder, burstName + '_merged_bayer'), mergedImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Dumped merged image')

	if params['writeReferenceImage']:
		with rawpy.imread(rawPathList[referenceIndex]) as rawReference:
			# process the raw reference image into a 8bit RGB matrix (demosaicking, white balance, gamma curve)
			paramsR = params['rawpyArgs'].copy()
			paramsR['output_bps'] = 8
			outputImage = rawReference.postprocess(**paramsR)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed ref. image')
		outputName = os.path.join(options['outputFolder'], burstName + '_reference.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
	if params['writeGammaReference']:
		with rawpy.imread(rawPathList[referenceIndex]) as rawReference:
			# process the raw reference image into a 8bit RGB matrix (demosaicking, white balance, gamma curve)
			outputImage = rawReference.postprocess(**params['rawpyArgs'])
		# normalize
		outputImage = outputImage / (2**16 - 1)
		outputImage = gammasRGB(outputImage, 'compress')
		# convert to 8 bit
		outputImage = convert8bit_(outputImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed ref. image w. gamma')
		outputName = os.path.join(options['outputFolder'], burstName + '_reference_gamma.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	if params['writeMergedImage']:
		# create a 8bit RGB non gamma corrected image from the merged bayer
		with rawpy.imread(rawPathList[referenceIndex]) as rawReference:
			# process the raw reference image into a 8bit RGB matrix (demosaicking, white balance, gamma curve)
			paramsR = params['rawpyArgs'].copy()
			paramsR['output_bps'] = 8
			# replace reference by merged bayer values
			rawReference.raw_image[:] = mergedImage[:]
			outputImage = rawReference.postprocess(**paramsR)
		# convert to 8 bit
		outputImage = convert8bit_(outputImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed merged image')
		# save image
		outputName = os.path.join(options['outputFolder'], burstName + '_merged.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
	if params['writeGammaMerged']:
		# create a 16bit RGB non gamma corrected image from the merged bayer
		with rawpy.imread(rawPathList[referenceIndex]) as rawReference:
			# process the raw reference image into a 8bit RGB matrix (demosaicking, white balance, gamma curve)
			rawReference.raw_image[:] = mergedImage[:]
			outputImage = rawReference.postprocess(**params['rawpyArgs'])
		# normalize
		outputImage = outputImage / (2**16 - 1)
		# apply gamma correction for visualization
		outputImage = gammasRGB(outputImage, 'compress')
		# convert to 8 bit
		outputImage = convert8bit_(outputImage)
		if verbose:
			currentTime = getTime(currentTime, ' -- Post-processed merged image w. gamma')
		# save image
		outputName = os.path.join(options['outputFolder'], burstName + '_merged_gamma.jpg')
		cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])

	return mergedImage


def centeredCosineWindow(x, windowSize=16):
	'''1D version of the modified raised cosine window (Section 4.4 of the IPOL article).
	It is centered and nonzero at x=0 and x=windowSize-1'''
	y = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * (x + 1 / 2.) / windowSize)
	return y


def cosineWindow2Dpatches(patches):
	'''Apply a 2D version of the modified raised cosine window
	to a set of overlapped patches to avoid discontinuities and edge artifacts
	(Section 4.4 of the IPOL article).
	Args:
		patches: 2D array of 2D patches (overlapped by half in each dimension)'''
	assert(len(patches.shape) == 4), 'not a 2D array of image patches'
	windowSize = patches.shape[-1]  # Assumes patches are square
	# Compute the attenuation window on 1 patch dimension
	lineWeights = centeredCosineWindow(np.arange(windowSize), windowSize).reshape(-1, 1).repeat(windowSize, 1)
	columnWeights = lineWeights.T
	# the 2D window is the product of the 1D window in both patches dimensions
	window = np.multiply(lineWeights, columnWeights)
	# Apply the attenuation cosine weighting to all patches
	return np.multiply(patches, window)


def cat2DPatches(patches):
	assert(len(patches.shape) == 4), "not a 2D array of 2D arrays"
	return np.concatenate(np.concatenate(patches, axis=1), axis=1)


def depatchifyOverlap(patches):
	'''recreates a single image out of a 2d arrangement
	of patches overlapped by half in each dimension
	'''
	assert(len(patches.shape) == 4), "not a 2D array of 2D patches"
	patchSize = patches.shape[-1]
	dp = patchSize // 2
	assert(patchSize == patches.shape[-2] and patchSize % 2 == 0), "function only supports square patches of even size"

	# separate the different groups of overlapped patches
	patchSet00 = patches[0::2, 0::2]  # original decomposition
	patchSet01 = patches[0::2, 1::2]  # straddled by patchSize/2 in horizontal axis
	patchSet10 = patches[1::2, 0::2]  # straddled by patchSize/2 in vertical axis
	patchSet11 = patches[1::2, 1::2]  # straddled by patchSize/2 half in both axes

	# recreate sub-images from the different patch groups
	imSet00 = cat2DPatches(patchSet00)
	imSet01 = cat2DPatches(patchSet01)
	imSet10 = cat2DPatches(patchSet10)
	imSet11 = cat2DPatches(patchSet11)

	# reconstruct final image by correctly adding sub-images
	reconstructedImage = np.zeros(((patches.shape[0] + 1) * dp, (patches.shape[1] + 1) * dp), dtype=imSet00.dtype)
	reconstructedImage[0 : imSet00.shape[0]     , 0 : imSet00.shape[1]     ]  = imSet00
	reconstructedImage[0 : imSet01.shape[0]     , dp: imSet01.shape[1] + dp] += imSet01
	reconstructedImage[dp: imSet10.shape[0] + dp, 0 : imSet10.shape[1]     ] += imSet10
	reconstructedImage[dp: imSet11.shape[0] + dp, dp: imSet11.shape[1] + dp] += imSet11
	return reconstructedImage


def patchesRMS(patches):
	'''Computes the Root-Mean-Square of a set of patches/tiles.
	Args:
		patches: nD array (n >= 3) of 2D patches
	'''
	assert len(patches.shape) >= 3, 'not an nD array of patches'
	# flatten each patch
	patches = np.reshape(patches, tuple(patches.shape[:-2] + (patches.shape[-2] * patches.shape[-1],)))
	return np.sqrt(np.mean(np.multiply(patches, patches), axis=-1))


@vectorize([complex128(complex128, complex128, float64), complex64(complex64, complex64, float32)], target='parallel')
def mergeWienerDFTPatches_(refPatches, altPatches, noiseVariance):
	# Keep the difference in memory for later use
	diff = refPatches - altPatches
	# Compute the squared absolute difference
	dist = diff.real**2 + diff.imag**2
	# Derive a shrinkage operator (Wiener filter variant)
	A = dist / (dist + noiseVariance)
	# The merge result is an interpolation of reference and alternate patches guided by operator A
	return altPatches + A * diff


def mergeWienerDFTPatches(referenceImagePatchesFFT, alternateImagePatchesFFT, noiseVariance, temporalFactor=8):
	'''Temporally denoise a pair of sets of 2D Frequency-domain patches
	Using a variant of the Wiener filter (Section 4.2 of the IPOL article).
	Args:
		referenceImagePatchesFFT: numpy array of 2D DFT patches of the reference image
		alternateImagePatchesFFT: numpy array of 2D DFT patches of the (aligned) alternate image
		noiseVariance: numpy array of the estimated noise variance for each patch
		temporalFactor: tuning factor that drives the compromise between ghosting and denoising.'''
	# scale the noise variance to match the scale of dSq
	patchSize = referenceImagePatchesFFT.shape[-1]
	noiseScaling = patchSize**2 * 1 / 4**2 * 2  # described in the paper
	# additional scaling according to temporal denoising strength tuning
	noiseScaling *= temporalFactor
	# Call the vectorized function
	mergedImagePatchesFFT = mergeWienerDFTPatches_(referenceImagePatchesFFT, alternateImagePatchesFFT, noiseScaling * noiseVariance)

	return mergedImagePatchesFFT


def temporalDenoisePairPatches(referenceImagePatches, alternateImagePatches, noiseVariance, temporalFactor=8, method='DFTWiener'):
	'''Temporally denoise two sets of 2D patches, and return a single set of denoised patches.'''
	if method == 'keepAlternate':
		# only return alternate image patches (e.g. to later average them)
		mergedImagePatches = alternateImagePatches
	elif method == 'pairedAverage':
		mergedImagePatches = np.multiply(0.5, referenceImagePatches + alternateImagePatches)
	elif method == 'DFTWiener':
		# referenceImagePatches then already contains 2D DFT of patches
		altPatchesFFT = torch.fft.fftn(torch.from_numpy(alternateImagePatches.astype(np.float32)), dim=(2, 3)).numpy()
		mergedImagePatches = mergeWienerDFTPatches(referenceImagePatches, altPatchesFFT, noiseVariance, temporalFactor)
	return mergedImagePatches


@vectorize([complex128(complex128, float64), complex64(complex64, float32)], target='parallel')
def spatialDenoisePatches_(patchesFFT, WienerCoeff):
	# Compute the magnitude of the patches coefficient
	dist = patchesFFT.real**2 + patchesFFT.imag**2
	# Wiener filtering
	return patchesFFT * dist / (dist + WienerCoeff)


def spatialDenoisePatches(patchesFFT, noiseVariance, spatialFactor):
	'''Spatially denoise a set of 2D Frequency-domain patches
	Using a variant of the Wiener filter (Section 4.3 of the IPOL article).
	Args:
		patchesFFT: numpy array of 2D DFT (temporally denoised) patches of the reference image.
		noiseVariance: numpy array of the (updated) estimated noise variance for each patch
		spatialFactor: tuning factor that drives the compromise between less residual noise and loss of high frequency content.'''
	assert(len(patchesFFT.shape) == 4), "not a 2D array of 2D patches"
	# Assumes square patches
	patchSize = patchesFFT.shape[-1]
	# create a patch of distance of spatial frequency module with respect to origin of spatial frequency axes
	# this patch will be used to filter higher frequency content more aggressively
	# WARNING: to avoid computing the FFT shift to the patches then FFT ishift,
	# This shift is applied directly to the distance
	rowDistances = (np.arange(patchSize) - patchSize / 2).reshape(-1, 1).repeat(patchSize, 1)
	columnDistances = rowDistances.T
	distancePatch = np.sqrt(rowDistances**2 + columnDistances**2)
	distPatchShift = np.fft.ifftshift(distancePatch, axes=(-2, -1))
	# Scale the noise variance to match the scale of np.abs(patchesFFT)**2
	noiseScaling = patchSize**2 * 1 / 4**2
	# Additional scaling according to spatial denoising strength tuning
	noiseScaling *= spatialFactor
	# Apply the Wiener filtering
	return spatialDenoisePatches_(patchesFFT, distPatchShift * noiseScaling * noiseVariance)


def getNoiseParams(tags, blackLevel, whiteLevel, params, options):
	'''retrieve noise curve parameters
	either from exif metadata, ISO and baseline values or from an input tuple
	(Section 4.1 of the IPOL article).
	Args:
		tags: dict of image metadata
		blackLevel: [int], per-channel black level of the reference image
		whiteLevel: int, reference image white level
		params: dict containing a parameter on how to retrieve the parameters
		options: dict containing options extracted from the script command
	'''
	if type(params['noiseCurve']) is tuple:
		lambdaS, lambdaR = params['noiseCurve']
	else:
		if params['noiseCurve'] == 'exifNoiseProfile':
			if options['verbose'] > 1:
				print('Looking for noise curve parameters in the NoiseProfile EXIF tag')
			if 'Image Tag 0xC761' in tags:
				noiseProfile = np.squeeze(tags['Image Tag 0xC761'].values)
				if len(noiseProfile) == 2:
					lambdaSn = noiseProfile[0]
					lambdaRn = noiseProfile[1]
				else:  # if noiseProfile has one value per CFA color
					assert len(noiseProfile) == 6
					assert noiseProfile[0] == noiseProfile[2] == noiseProfile[4], 'NoiseProfile tag is different for each channel'
					assert noiseProfile[1] == noiseProfile[3] == noiseProfile[5], 'NoiseProfile tag is different for each channel'
					lambdaSn = noiseProfile[0]
					lambdaRn = noiseProfile[1]
			else:
				ISO = 100
				if 'Image ISOSpeedRatings' in tags:
					ISO = tags['Image ISOSpeedRatings'].values[0]
				elif 'EXIF ISOSpeedRatings' in tags:
					ISO = tags['EXIF ISOSpeedRatings'].values[0]
				if ISO == 0:
					ISO = 100  # some images can have incorrect ISO data
				if options['verbose'] > 1:
					print('NoiseProfile tag not found. Computing lambdaS and lambdaR from ISO({}) and baseline values'.format(ISO))
				baselineLambdaS, baselineLambdaR = 3.24 * 10**(-4), 4.3 * 10**(-6)  # average values normalized at ISO 100 from images with the NoiseProfile tag
				lambdaSn = ISO / 100 * baselineLambdaS
				lambdaRn = (ISO / 100)**2 * baselineLambdaR
		elif params['noiseCurve'] == 'exifISO':
			baselineNoise = tags['Image Tag 0xC62B'].values[0] if 'Image Tag 0xC62B' in tags else 1
			ISO = 100
			if 'Image ISOSpeedRatings' in tags:
				ISO = tags['Image ISOSpeedRatings'].values[0]
			elif 'EXIF ISOSpeedRatings' in tags:
				ISO = tags['EXIF ISOSpeedRatings'].values[0]
			if ISO == 0:
				ISO = 100  # some images can have incorrect ISO data
			if options['verbose'] > 1:
				print('Computing lambdaS and lambdaR from ISO({}) and baseline values'.format(ISO))
			baselineLambdaS, baselineLambdaR = 3.24 * 10**(-4), 4.3 * 10**(-6)  # average values normalized at ISO 100 from images with the NoiseProfile tag
			lambdaSn = ISO / 100 * baselineLambdaS
			lambdaRn = (ISO / 100)**2 * baselineLambdaR
		# lambdaSn and lambdaRn are noise curve parameters for an image with values between 0 and 1
		# unnormalize: var(k*x) = k**2*var(x) = k**2*(lambdaSn*x+lambdaRn) = k*lambdaSn*(k*x) + k**2*lambdaRn
		bL = np.min(blackLevel)  # per channel difference too small to matter
		lambdaS, lambdaR = lambdaSn * (whiteLevel - bL), lambdaRn * (whiteLevel - bL)**2
	if options['verbose'] > 2:
		print('Noise curve parameters: lambdaS={}, lambdaR={}'.format(lambdaS, lambdaR))
	return lambdaS, lambdaR


def mergeChannelHdrplus(referenceChannel, alignedChannelTiles, lambdaS, lambdaR, params, options):
	'''perform per-channel, tile-based, pairwise temporal denoising
	as defined in Algorithm 2 of Section 4.2 of the IPOL article.'''
	# Initialization
	currentTime, verbose = time.time(), options['verbose'] > 3
	referenceChannelTiles = alignedChannelTiles[0]
	noiseVariance = []
	if verbose:
		currentTime = getTime(currentTime, ' ---- Initialization')
	if params['method'] == 'DFTWiener':
		# compute noise model of reference image
		# signal level is considered constant for each patch
		signalLevel = patchesRMS(referenceChannelTiles)
		if verbose:
			currentTime = getTime(currentTime, ' ---- Get signal level')
		# noiseVariance = lambdaS * signalLevel + lambdaR
		# compute the per-patch variance value (same value for all pixels of the patch)
		noiseVariance = (np.multiply(lambdaS, signalLevel[..., np.newaxis, np.newaxis]) + lambdaR).repeat(params['patchSize'], axis=-2).repeat(params['patchSize'], axis=-1)
		if verbose:
			currentTime = getTime(currentTime, ' ---- Get noise variance')
		# compute 2D DFT of reference image tiles
		referenceChannelTiles = torch.fft.fftn(torch.from_numpy(referenceChannelTiles.astype(np.float32)), dim=(2, 3)).numpy()
		if verbose:
			currentTime = getTime(currentTime, ' ---- Compute 2D DFT')

	# pairwise merging of reference image channel with itself = reference image channel
	mergedPairsTilesSum = referenceChannelTiles.copy()
	if verbose:
			currentTime = getTime(currentTime, ' ---- Copy reference')

	if options['temporalFactor'] == 0:
		# no temporal denoising
		mergedChannelTiles = mergedPairsTilesSum
	else:
		# pairwise merging of all other images with reference image
		for i in range(1, len(alignedChannelTiles)):
			mergedPairsTilesSum += temporalDenoisePairPatches(referenceChannelTiles, alignedChannelTiles[i], noiseVariance, options['temporalFactor'], params['method'])
		if verbose:
				currentTime = getTime(currentTime, ' ---- Temporal denoise pair patches')
		# final merge = mean of all pairwise merges
		mergedChannelTiles = mergedPairsTilesSum / len(alignedChannelTiles)
	if verbose:
			currentTime = getTime(currentTime, ' ---- Average tiles')

	if params['method'] == 'DFTWiener':
		if options['spatialFactor'] > 0:
			# update noise variance estimation after temporal denoising
			noiseVariance /= len(alignedChannelTiles)
			# perform fast spatial denoising using a Wiener filter
			mergedChannelTiles = spatialDenoisePatches(mergedChannelTiles, noiseVariance, options['spatialFactor'])
			if verbose:
				currentTime = getTime(currentTime, ' ---- Spatial denoise')

		mergedChannelTiles = torch.fft.ifftn(torch.from_numpy(mergedChannelTiles), dim=(2, 3)).real.numpy()
		if verbose:
			currentTime = getTime(currentTime, ' ---- DFT inverse')

	# apply a cosine window function to the overlapping patches to avoid edge artifacts
	mergedChannelTiles = cosineWindow2Dpatches(mergedChannelTiles)
	if verbose:
			currentTime = getTime(currentTime, ' ---- Cosine window')

	# reconstruct an unsigned integer image from the patches
	mergedChannel = depatchifyOverlap(mergedChannelTiles).astype(referenceChannel.dtype)
	if verbose:
			currentTime = getTime(currentTime, ' ---- Depatchify')

	return mergedChannel


def mergeHdrplus(referenceImage, alignedTiles, padding, tags, blackLevel, whiteLevel, params, options):
	'''Implements the Fourier Tile-based Merging as described in Section 4 of the IPOL artice.
	Args:
		referenceImage: 2D array of the reference (Bayer) Image
		alignedTiles: ndarray of tiles aligned after motion estimation
		padding: tuple of padding applied to all images (will need to be removed)
		tags: dict of image metadata
		blackLevel: [int], per-channel black level of the reference image
		whiteLevel: int, reference image white level
		params: dict containing both algorithm parameters and output choices
		options: dict containing options extracted from the script command (input/output path, mode, verbose)
	'''
	# Initialization
	currentTime, verbose = time.time(), options['verbose'] > 2
	# perform burst merging as pairwise merging of reference image and each alternate image
	tileSize = alignedTiles.shape[3]
	h, w = (alignedTiles.shape[1] + 1) * tileSize // 2, (alignedTiles.shape[2] + 1) * tileSize // 2
	mergedImage = np.empty((h, w), dtype=referenceImage.dtype)
	# get noise curve parameters
	lambdaS, lambdaR = getNoiseParams(tags, blackLevel, whiteLevel, params, options)
	if verbose:
		currentTime = getTime(currentTime, ' --- Initialization')

	# work separately on each channel of the Bayer image
	for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
		# for each channel, compute burst merging as described in the HDR+ paper
		mergedImage[di::2, dj::2] = mergeChannelHdrplus(referenceImage[di::2, dj::2], alignedTiles[..., di::2, dj::2], lambdaS, lambdaR, params, options)
		if options['verbose'] > 3:
			currentTime = getTime(currentTime, ' --- Channel %d/4 merged' % c)

	return mergedImage[padding[0]:h - padding[1], padding[2]:w - padding[3]]  # padding discarded here
