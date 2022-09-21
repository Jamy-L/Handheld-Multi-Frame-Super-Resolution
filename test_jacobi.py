# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:59:22 2022

@author: jamyl
"""




import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
from time import time
import cupy as cp
from tqdm import tqdm

import math
from torch import from_numpy

from linalg import solve_6x6_krylov
from scipy.stats import ortho_group

D = np.random.random(6)*10+5
A = np.zeros((6,6))
np.fill_diagonal(A, D)
P = ortho_group.rvs(dim = 6)
A = P.transpose()@A@P # A is definite positive
X = np.random.random(6)
B = A@X
#%%
print("conditionnement : ", np.linalg.norm(A)*np.linalg.norm(np.linalg.inv(A)))

def dummy_krylov(A, B, X):
    r = B - A@X
    d = r.copy()
    delta = np.linalg.norm(r)**2

    for k in range(6):
        delta_p = d@A@d
        lambda_ = delta/delta_p
        X += lambda_*d
        r -= lambda_*A@d
        delta_new = np.linalg.norm(r)**2
        beta = delta_new/delta
        d = r+ beta*d
        delta = delta_new
    


@cuda.jit
def solve(A,B, Xt):
    solve_6x6_krylov(A, B, Xt, 5)



Xt = cuda.to_device(np.zeros(6))
Xt2 = np.zeros(6)

solve[(1), (7, 7)](A,B,Xt)

Xt = Xt.copy_to_host()
dummy_krylov(A, B, Xt2)

