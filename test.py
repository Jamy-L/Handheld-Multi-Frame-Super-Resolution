# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:16:53 2022

@author: jamyl
"""

from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, prange, int8
import numpy as np
from time import time
import cupy as cp


@guvectorize(['(float64[:,:,:], float64[:,:], float64[:,:])'],
             '(l, m, n), (l, m) -> (l, m)')  # target="cuda")
def solve(A, B, C):
    l, m, n = A.shape
    for i in range(l):
        C[l] = np.linalg.solve(A[i], B[i])


A = np.random.random((5000000, 2, 2))
B = np.random.random((5000000, 2))

t1 = time()
Ac = cp.array(A)
Bc = cp.array(B)
cp.linalg.solve(Ac, Bc)
print(time() - t1)


t1 = time()
C = solve(A, B)
print(time()-t1)
