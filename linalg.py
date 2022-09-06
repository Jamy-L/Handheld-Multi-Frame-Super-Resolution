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
def invert_2x2(M, M_i):
    """
    inverts the 2x2 M array

    Parameters
    ----------
    M : Array[2, 2]
        Array to invert
    M_i : Array[2, 2]

    Returns
    -------
    None.

    """
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    M_i[0,0] = M[1,1]/det
    M_i[0, 1] = -M[0, 1]/det
    M_i[1, 0] = -M[1, 0]/det
    M_i [1, 1] = M[0, 0]/det
    
@cuda.jit(device=True)
def quad_mat_prod(A, X):
    return A[0, 0]*X[0]*X[0] + X[0]*X[1]*(A[0, 1] + A[1, 0]) + A[1, 1]*X[1]*X[1]

@cuda.jit(device=True)
def get_real_polyroots_2(a, b, c, roots):
    """
    Returns the two roots of the polynom a*X^2 + b*X + c = 0 for a, b and c
    real numbers. The function only returns real roots : make sure they exist
    before calling the function. l[0] contains the root with the biggest module
    and l[1] the smallest


    Parameters
    ----------
    a : float

    b : float

    c : float
    
    roots : Array[2]

    Returns
    -------
    None

    """

    delta = b*b - 4*a*c

    if delta >= 0:
        r1 = (-b+sqrt(delta))/(2*a)
        r2 = (-b-sqrt(delta))/(2*a)
        if abs(r1) >= abs(r2) :
            roots[0] = r1
            roots[1] = r2
        else:
            roots[0] = r2
            roots[1] = r1
    else:
        # Nan
        return 1/0

@cuda.jit(device=True)
def get_eighen_val_2x2(M, l):
    a = 1
    b = -(M[0,0] + M[1, 1])
    c = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    get_real_polyroots_2(a, b, c, l)

# TODO Im not sure if its the best thing to do
@cuda.jit(device=True)
def get_eighen_vect_2x2(M, l, e1, e2):
    """
    return the eighen vector with norm 1 for eighen values l
    Me1 = l1e1 ; Me2 = l2e2

    Parameters
    ----------
    M : Array[2,2]
      Array for which eighen values are to be determined
    l : Array[2, 2]
        Eighenvalues
    e1, e2 : Array[2]
        Computed eighen vectors

    Returns
    -------
    None.

    """
    # 2x2 algorithm : https://en.wikipedia.org/wiki/Eigenvalue_algorithm (9 August 2022 version)
    if M[0, 1] == 0 and M[1, 0] ==0 and M[0,0] == M[1, 1]:
        # M is multiple of identity, picking 2 ortogonal eighen vectors.
        e1[0] = 1; e1[1] = 0
        e2[0] = 0; e2[0] = 1
        
    else:
        e1[0] = M[0, 0] - l[1]; e1[1] = M[1,0]
        e2[0] = M[0, 0] - l[0]; e2[0] = M[1,0]
        
    
    
    
@cuda.jit(device=True)
def get_eighen_elmts_2x2(M, l, e1, e2):
    
    get_eighen_val_2x2(M, l)
    get_eighen_vect_2x2(M, l, e1, e2)
    
    
    
    
    
    
    
    