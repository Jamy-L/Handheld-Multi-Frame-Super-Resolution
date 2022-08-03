"""Generic utility functions.
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
"""

# imports
import time
import numpy as np


def getTime(currentTime, labelName, printTime=True, spaceSize=50):
	'''Print the elapsed time since currentTime. Return the new current time.'''
	if printTime:
		print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.time() - currentTime) * 1000, 2), 'milliseconds')
	return time.time()


def getSigned(array):
	'''Return the same array, casted into a signed equivalent type.'''
	# Check if it's an unssigned dtype
	dt = array.dtype
	if dt == np.uint8:
		return array.astype(np.int16)
	if dt == np.uint16:
		return array.astype(np.int32)
	if dt == np.uint32:
		return array.astype(np.int64)
	if dt == np.uint64:
		return array.astype(np.int)

	# Otherwise, the array is already signed, no need to cast it
	return array


def isTypeInt(array):
	'''Check if the type of a numpy array is an int type.'''
	return array.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.uint, np.int]
