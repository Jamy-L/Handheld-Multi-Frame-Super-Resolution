"""Script to run the hdrplus implementation on a folder containing multiple bursts.
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
import time
import glob
# custom package imports
from package.algorithm.hdrplus import hdrplusPipeline
from package.algorithm.genericUtils import getTime
from package.algorithm.params import getParams
from runHdrplus import *


def runHdrplus(options):
	'''Main function to process bursts.'''
	# For convenience
	currentTime, verbose = time.time(), options['verbose'] > 0

	burstList = glob.glob(os.path.join(options['inputFolder'], '*'))
	# sort the burst list in ascending order for better repeatability
	burstList.sort()
	# Get the parameters that correspond to the selected mode
	params = getParams(options['mode'])

	# Process all the bursts
	for burstNumber, burstName in enumerate(burstList):
		if verbose:
			print("=" * (20 + len(str(burstNumber + 1) + str(len(burstList))) + len(burstName)))
			print("Processing burst {}/{}: {}".format(burstNumber + 1, len(burstList), burstName))
			print("=" * (20 + len(str(burstNumber + 1) + str(len(burstList))) + len(burstName)))
		# Run the pipeline on the burst
		hdrplusPipeline(os.path.join(options['inputFolder'], burstName), params, options)
		if verbose:
			currentTime = getTime(currentTime, ' - Burst processed')


if __name__ == "__main__":

	# Get the arguments
	options = getOptions(getParsers())
	# Get the parameters that correspond to the selected mode
	params = getParams(options['mode'])

	# Create output folder if needed
	if not os.path.isdir(options['outputFolder']):
		os.makedirs(options['outputFolder'])

	# For convenience
	currentTime, verbose = time.time(), options['verbose'] > 0

	burstList = glob.glob(os.path.join(options['inputFolder'], '*'))
	# sort the burst list in ascending order for better repeatability
	burstList.sort()

	# Process all the bursts
	for burstNumber, burstName in enumerate(burstList):
		if verbose:
			print("=" * (20 + len(str(burstNumber + 1) + str(len(burstList))) + len(burstName)))
			print("Processing burst {}/{}: {}".format(burstNumber + 1, len(burstList), burstName))
			print("=" * (20 + len(str(burstNumber + 1) + str(len(burstList))) + len(burstName)))
		# Run the pipeline on the burst
		hdrplusPipeline(os.path.join(options['inputFolder'], burstName), params, options)
		if verbose:
			currentTime = getTime(currentTime, ' - Burst processed')

	# Total elapsed time
	getTime(currentTime, 'Total process')
