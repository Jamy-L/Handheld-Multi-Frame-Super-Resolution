# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:56:46 2022

@author: jamyl
"""

import numpy as np
from numba import cuda, float32, float64
import time

threads = 32
threads_sq = threads**2
blocks = 50000

A = np.random.random((threads,threads, blocks))
B = np.random.random((threads,threads))
cu_C = cuda.device_array(blocks, np.float64)
cu_D = cuda.device_array(blocks, np.float64)

@cuda.jit
def fancy_add(A, B, C):
    x, y = cuda.threadIdx.x, cuda.threadIdx.y
    z = cuda.blockIdx.x
    
    
    tx = int(x + y*threads)
    
    s = cuda.shared.array(threads_sq, float64)
    dif = A[x, y, z] - B[x, y]
    s[tx] = dif*dif
    
    
    # reduction
    step = 1
    for reduc in range(7):
        
        cuda.syncthreads()
        if tx%(2*step) == 0:
            s[tx] += s[tx + step]
        
        step *= 2
    
    cuda.syncthreads()
    if x == 0 and y==0:
        C[z] = s[0]

@cuda.jit
def lazy_add(A, B, C):
    x, y = cuda.threadIdx.x, cuda.threadIdx.y
    z = cuda.blockIdx.x
    
    
    s = cuda.shared.array(1, float64)
    dif = A[x, y, z] - B[x, y]
    cuda.atomic.add(s, 0, dif*dif)
    
    cuda.syncthreads()
    if x == 0 and y==0:
        C[z] = s[0]
        
cu_A = cuda.to_device(A)
cu_B = cuda.to_device(B)

cuda.synchronize()
t1 = time.perf_counter()
fancy_add[(blocks), (threads, threads)](cu_A, cu_B, cu_C)
cuda.synchronize()
print("fancy : ", time.perf_counter() - t1)

cuda.synchronize()
t1 = time.perf_counter()
lazy_add[(blocks), (threads, threads)](cu_A, cu_B, cu_D)
cuda.synchronize()
print("lazy : ", time.perf_counter() - t1)




C = cu_C.copy_to_host()
gt = np.sum(A+B[:,:, np.newaxis], axis=(0, 1))

