# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:41:58 2022

@author: jamyl
"""

import numpy as np
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda
from math import sqrt


@cuda.jit(device=True)
def solve_2x2(A, B, X):
    """
    Cuda function for resolving the 2x2 system A*X = B
    by using the analytical formula

    Parameters
    ----------
    A : Array[2,2]

    B : Array[2]

    Returns
    -------
    None

    """
    det_A = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]

    X[0] = (A[1, 1]*B[0] - A[0, 1]*B[1])/det_A
    X[1] = (A[0, 0]*B[1] - A[1, 0]*B[0])/det_A


@cuda.jit(device=True)
def sym_quad():
    pass


@cuda.jit(device=True)
def get_real_polyroots_2(a, b, c, roots):
    """
    Returns the two roots of the polynom a*X^2 + b*X + c = 0 for a, b and c
    real numbers. The function only returns real roots : make sure they exist
    before calling the function


    Parameters
    ----------
    a : float

    b : float

    c : float
    
    root : Array[2]

    Returns
    -------
    None

    """

    delta = b*b - 4*a*c

    if delta >= 0:
        roots[0] = (-b-sqrt(delta))/(2*a)
        roots[1] = (-b+sqrt(delta))/(2*a)

@cuda.jit(device=True)
def get_eighen_val_2x2(M, l):
    a = 1
    b = -(M[0,0] + M[1, 1])
    c = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    get_real_polyroots_2(a, b, c, l)

# TODO Im not sure if its the best thing to do
@cuda.jit(device=True)
def get_eighen_vect_2x2(M, l, X):
    """
    return the eighen vector with norm 1 for eighen value l
    MX = lX

    Parameters
    ----------
    M : Array[2,2]
      Array for which eighen values are to be determined
    l : float64
        eighenvalue
    X : Array[2]
        Solution

    Returns
    -------
    None.

    """
    X[1] = 1/sqrt((M[0,1]/(M[0,0] - l))*(M[0,1]/(M[0,0] - l)) + 1)
    X[0] = - X[1] * M[0, 1]/(M[0,0] - l)
    
    
@cuda.jit(device=True)
def get_eighen_elmts_2x2(M, l, e1, e2):
    
    get_eighen_val_2x2(M, l)
    
    e1 = get_eighen_vect_2x2(M, l[0], e1)
    e2 = get_eighen_vect_2x2(M, l[1], e2)
    
    
    
    
    
    
    
    
    