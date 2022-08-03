"""Core hdrplus function that runs over a burst.
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
import glob
import time
import rawpy
import exifread
import numpy as np
# package-specific imports (Package named 'package.algorithm')
from .genericUtils import getTime
from .alignment import selectReference, alignBurst
from .merging import mergeBurst
from .finishing import finish


def hdrplusPipeline(burstPath, params, options):
	'''This function encompasses the whole HDR+ pipeline as described in the article.
	Args:
		bursPath: str, path to the folder containing the .dng files of the burst
		params: dict containing both algorithm parameters and output choices
		options: dict containing options extracted from the script command (input/output path, mode, verbose)
	'''
	# For convenience
	currentTime, verbose = time.time(), options['verbose'] > 1
	rawBayers, alignedTiles, mergedBayer = [], [], []

	# Get the list of raw images in the burst path
	rawPathList = glob.glob(os.path.join(burstPath, '*.dng'))
	# sort the raw list in ascending order to avoid errors in reference frame selection
	rawPathList.sort()
	assert rawPathList != [], 'At least one raw .dng file must be present in the burst folder.'
	# Read the raw bayer data from the DNG files
	for rawPath in rawPathList:
		with rawpy.imread(rawPath) as rawObject:
			rawBayers.append(rawObject.raw_image.copy())  # copy otherwise image data is lost when the rawpy object is closed

	# Reference image selection
	refIdx = selectReference(burstPath, rawBayers, options)
	if verbose:
		currentTime = getTime(currentTime, ' -- Read raw files')
	# get reference image metadata
	with open(rawPathList[refIdx], 'rb') as rawFile:
		tags = exifread.process_file(rawFile)
	with rawpy.imread(rawPathList[refIdx]) as refRawpy:
		blackLevel = refRawpy.black_level_per_channel.copy()
		whiteLevel = refRawpy.white_level

	if options['mode'] == 'full':
		# Burst alignment/registration
		alignedTiles, padding = alignBurst(burstPath, rawPathList, rawBayers, refIdx, params['alignment'], options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Aligned burst')
		# Burst merging
		mergedBayer = mergeBurst(burstPath, rawPathList, rawBayers, refIdx, alignedTiles, padding, tags, blackLevel, whiteLevel, params['merging'], options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Merged burst')
		# finishing
		finish(burstPath, rawPathList, refIdx, mergedBayer, params['finishing'], options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Finished + wrote output files')

	elif options['mode'] == 'align':
		# Burst alignment/registration
		alignedTiles, padding = alignBurst(burstPath, rawPathList, rawBayers, refIdx, params['alignment'], options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Aligned burst')

	elif options['mode'] == 'merge':
		# folder should only contain one .dng file we will use when writing output files
		assert len(rawPathList) == 1, 'Folder of aligned burst contains more than one .dng file'
		# padding is stored in a .npy file
		paddingPath = glob.glob(os.path.join(burstPath, '*padding*.npy'))
		assert paddingPath != [], 'No .npy files corresponding to padding found in the burst folder.'
		padding = np.load(paddingPath[0])
		# tiles of aligned burst images are stored as .npy files
		alignedTilesPathList = glob.glob(os.path.join(burstPath, '*aligned_tiles*.npy'))
		assert alignedTilesPathList != [], 'No .npy file corresponding to aligned tiles found in the burst folder.'
		alignedTiles = np.load(alignedTilesPathList[0])
		# Burst merging
		mergedBayer = mergeBurst(burstPath, rawPathList, rawBayers, refIdx, alignedTiles, padding, tags, blackLevel, whiteLevel, params['merging'], options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Merged burst')

	elif options['mode'] == 'finish':
		# folder should only contain one .dng file we will use when writing output files
		assert len(rawPathList) >= 1, 'Folder must contain at least one .dng file for finishing'
		# burst images are already aligned and stored as single channel .png files
		mergedBayerFile = glob.glob(os.path.join(burstPath, '*merged_bayer*.npy'))
		assert mergedBayerFile != [], "No .npy file containing 'merged_bayer' found in the burst folder."
		mergedBayer = np.load(mergedBayerFile[0])
		# finishing
		finish(burstPath, rawPathList, refIdx, mergedBayer, params['finishing'], options)
		if verbose:
			currentTime = getTime(currentTime, ' -- Finished pipeline')

	return
