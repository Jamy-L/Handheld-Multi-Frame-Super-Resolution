# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""
import time
import numpy as np

from numba import uint8, uint16, float32, float64, cuda

def getTime(currentTime, labelName, printTime=True, spaceSize=50):
	'''Print the elapsed time since currentTime. Return the new current time.'''
	if printTime:
		print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.time() - currentTime) * 1000, 2), 'milliseconds')
	return time.time()

def isTypeInt(array):
	'''Check if the type of a numpy array is an int type.'''
	return array.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.uint, np.int]


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


@cuda.jit(device=True)
def clamp(x, min_, max_):   
    if x < min_ :
        return min_
    elif x > max_:
        return max_
    else:
        return x