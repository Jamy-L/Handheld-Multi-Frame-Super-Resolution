# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 10:43:19 2022

@author: jamyl
"""

import numpy as np
from numba import cuda, float32
import time

threads = 16



@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B."""
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B using CUDA shared memory.

    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
    """
    TPB = cuda.blockDim[0]
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp
    
        
A = np.random.random((40*threads, 40*threads))
cuda_A = cuda.to_device(A)

B = np.random.random((40*threads, 40*threads))
cuda_B = cuda.to_device(B)

C = cuda.device_array(A.shape, np.float32)

cuda.synchronize()
t1 = time.time()
matmul[(40, 40), (threads, threads)](cuda_A, cuda_B, C)
cuda.synchronize()
print("naive : ", time.time()-t1)

cuda.synchronize()
t1 = time.time()
matmul[(40, 40), (threads, threads)](cuda_A, cuda_B, C)
cuda.synchronize()
print("fancy : ", time.time()-t1)