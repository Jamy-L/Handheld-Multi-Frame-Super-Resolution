# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:41:58 2022

@author: jamyl
"""

import numpy as np
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda
from math import sqrt


@cuda.jit(device=True)
def solve_2x2(A, B):
    """
    Cuda function for resolving the 2x2 system A*X = B
    by using the analytical formula

    Parameters
    ----------
    A : Array[2,2]

    B : Array[2]

    Returns
    -------
    X : Array[2]
        Solution of the system.

    """
    X = cuda.device_array(shape=2)
    det_A = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]

    X[0] = (A[1, 1]*B[0] - A[0, 1]*B[1])/det_A
    X[1] = (A[0, 0]*B[1] - A[1, 0]*B[0])/det_A

    return X


@cuda.jit(device=True)
def real_polyroots_2(a, b, c):
    """
    Returns the two roots of the polynom a*X^2 + b*X + c = 0 for a, b and c
    real numbers. The function only returns real roots : make sure they exist
    before calling the function


    Parameters
    ----------
    a : float

    b : float

    c : float

    Returns
    -------
    X : Array[2]
        solutions [x_1, x_2]

    """

    delta = b*b - 4*a*c
    X = cuda.device_array(shape=2)

    if delta >= 0:
        X[0] = (-b-sqrt(delta))/(2*a)
        X[1] = (-b+sqrt(delta))/(2*a)
    return X
