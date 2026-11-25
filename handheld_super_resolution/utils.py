"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""

import time

import numpy as np
import torch as th
import torch.fft

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


def timer(func, enabled, start_s=None, end_s=None, spaceSize=50):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        if start_s is not None:
            print(start_s)

        out = func(*args, **kwargs)

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
