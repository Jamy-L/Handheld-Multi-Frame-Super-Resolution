# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""
import time
from math import cos, pi

import numpy as np
from numba import float32, float64, complex64, cuda
import torch as th
import torch.fft


DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
EPSILON = 1e-6

DEFAULT_THREADS = 16


def getTime(currentTime, labelName, printTime=True, spaceSize=50):
	'''Print the elapsed time since currentTime. Return the new current time.'''
	if printTime:
		print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.perf_counter() - currentTime) * 1000, 2), 'milliseconds')
	return time.perf_counter()

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


def divide(num, den):
    """
    Performs num = num/den

    Parameters
    ----------
    num : device array[ny, nx, n_channels]
        DESCRIPTION.
    den : device array[ny, nx, n_channels]
        DESCRIPTION.


    """
    assert num.shape == den.shape
    n_channels = num.shape[-1]
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = int(np.ceil(num.shape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(num.shape[0]/threadsperblock[0]))
    blockspergrid_z = n_channels
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_divide[blockspergrid, threadsperblock](num, den)

@cuda.jit
def cuda_divide(num, den):
    x, y, c = cuda.grid(3)
    if 0 <= x < num.shape[1] and 0 <= y < num.shape[0] and 0 <= c < num.shape[2]:
        num[y, x, c] = num[y, x, c]/den[y, x, c]
        
def add(A, B):
    """
    performs A += B for 2d arrays

    Parameters
    ----------
    A : device_array[ny, nx]

    B : device_array[ny, nx]
        

    Returns
    -------
    None.

    """
    assert A.shape == B.shape
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = int(np.ceil(A.shape[1]/threadsperblock[1]))
    blockspergrid_y = int(np.ceil(A.shape[0]/threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_add[blockspergrid, threadsperblock](A, B)

@cuda.jit
def cuda_add(A, B):
    x, y = cuda.grid(2)
    if 0 <= x < A.shape[1] and 0 <= y < A.shape[0]:
        A[y, x] += B[y, x]
    

