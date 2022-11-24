# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""
import time
from math import cos, pi

import numpy as np
from numba import uint8, uint16, float32, float64, complex64, cuda
import torch as th
import torch.fft


DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
EPSILON = 1e-6


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
	# Check if it's an unsigned dtype
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
    return min(max_, max(min_, x))

def mse(im1, im2):
    return np.linalg.norm(im1 - im2) / np.prod(im1.shape)


# TODO This is costy but numpy does not make it better ... Maybe saving a table beforehand and reading is better
@cuda.jit(device = True)
def hann(i, j, tile_size):
    return (0.5 + 0.5*cos(2*pi*i/tile_size)) * (0.5 + 0.5*cos(2 * pi*j/tile_size))

@cuda.jit(device = True)
def hamming(i, j, tile_size):
    return (0.54 + 0.46*cos(2*pi*i/tile_size)) * (0.54 + 0.46*cos(2*pi*j/tile_size))


# for debugging and testing only, this is probably dirty and unoptimised
def crop(array, crop_str, axis):
    """
    

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    crop_str : str "[ymin:ymax, xmin:xmax]"
        area to be cropped
    axis : (y, x)
        axis of y and x in the arrau
        

    Returns
    -------
    a : cropped array
        DESCRIPTION.

    """
    crop_y, crop_x = crop_str[1:-1].replace(' ', '').split(",")
    crop_y_min, crop_y_max= crop_y.split(':')
    crop_x_min, crop_x_max= crop_x.split(':')
    
    crop_y_min = int(crop_y_min)
    crop_y_max = int(crop_y_max)
    crop_x_min = int(crop_x_min)
    crop_x_max = int(crop_x_max)
        
    a = array.take(indices=range(crop_y_min, crop_y_max), axis=axis[0])
    a = a.take(indices=range(crop_x_min, crop_x_max), axis=axis[1])
    return a