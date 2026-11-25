"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""

import time

import numpy as np
import torch as th
import torch.fft
from numba import cuda, float32

DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
EPSILON_DIV = 1e-10

DEFAULT_THREADS = 16
FINAL_TILE_SIZE = 16
FINAL_TILE_STRIDE = 8
FINAL_TILE_ORIGIN = FINAL_TILE_SIZE // 2 - FINAL_TILE_SIZE // 2


def getTime(currentTime, labelName, printTime=True, spaceSize=50):
    """Print the elapsed time since currentTime. Return the new current time."""
    if printTime:
        print(
            labelName,
            " " * (spaceSize - len(labelName)),
            ": ",
            round((time.perf_counter() - currentTime) * 1000, 2),
            "milliseconds",
        )
    return time.perf_counter()


def isTypeInt(array):
    """Check if the type of a numpy array is an int type."""
    return array.dtype in [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint,
        np.int,
    ]


def getSigned(array):
    """Return the same array, casted into a signed equivalent type."""
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


def mse(im1, im2):
    return np.linalg.norm(im1 - im2) / np.prod(im1.shape)


def timer(func, enabled, start_s=None, end_s=None, spaceSize=50):
    def wrapper(*args, **kwargs):
        cuda.synchronize()
        t1 = time.perf_counter()
        if start_s is not None:
            print(start_s)

        out = func(*args, **kwargs)

        cuda.synchronize()

        if end_s is not None:
            print(
                end_s,
                " " * (spaceSize - len(end_s)),
                ": ",
                round((time.perf_counter() - t1) * 1000, 2),
                "milliseconds",
            )

        return out

    if enabled:
        return wrapper
    else:
        return func
