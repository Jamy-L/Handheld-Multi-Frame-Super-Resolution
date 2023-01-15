# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:41:58 2022

@author: jamyl
"""

import math

from numba import cuda

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
    det_i = 1/(M[0,0]*M[1,1] - M[0,1]*M[1,0])
    if math.isinf(det_i):
        M_i[0, 0] = 1
        M_i[0, 1] = 0
        M_i[1, 0] = 0
        M_i[1, 1] = 1
    else:
        M_i[0, 0] = M[1,1]*det_i
        M_i[0, 1] = -M[0, 1]*det_i
        M_i[1, 0] = -M[1, 0]*det_i
        M_i[1, 1] = M[0, 0]*det_i
    
@cuda.jit(device=True)
def quad_mat_prod(A, X1, X2):
    """
    With X = [X1, X2], performs the quadratique form :
        X.transpose() @ A @ X

    Parameters
    ----------
    A : device Array[2, 2]
    X1 : float
    X2 : float

    Returns
    -------
    y : float

    """
    y = A[0, 0]*X1*X1 + X1*X2*(A[0, 1] + A[1, 0]) + A[1, 1]*X2*X2
    return y

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
        r1 = (-b+math.sqrt(delta))/(2*a)
        r2 = (-b-math.sqrt(delta))/(2*a)
        if abs(r1) >= abs(r2) :
            roots[0] = r1
            roots[1] = r2
        else:
            roots[0] = r2
            roots[1] = r1
    else:
        # Nan
        roots[0] = 0/0
        roots[1] = 0/0

@cuda.jit(device=True)
def get_eighen_val_2x2(M, l):
    a = 1
    b = -(M[0,0] + M[1, 1])
    c = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    get_real_polyroots_2(a, b, c, l)

@cuda.jit(device=True)
def get_eighen_vect_2x2(M, l, e1, e2):
    """
    return the eighen vectors with norm 1 for the eighen values l
    M.e1 = l1.e1 ; M.e2 = l2.e2

    Parameters
    ----------
    M : Array[2,2]
      Real Symmetric array for which eighen values are to be determined
    l : Array[2]
    e1, e2 : Array[2]
        sorted Eigenvalues
    e1, e2 : Array[2, 2]
        Computed orthogonal and normalized eigen vectors

    Returns
    -------
    None.

    """
    # 2x2 algorithm : https://en.wikipedia.org/wiki/Eigenvalue_algorithm (9 August 2022 version)
    if M[0, 1] == 0 and M[0, 0] == M[1, 1]:
        # M is multiple of identity, picking 2 ortogonal eighen vectors.
        e1[0] = 1; e1[1] = 0
        e2[0] = 0; e2[0] = 1
        
    else:
        # averaging 2 for increased reliability
        e1[0] = M[0, 0] + M[0, 1] - l[1] 
        e1[1] = M[1, 0] + M[1, 1] - l[1] 
        
        if e1[0] == 0:
            e1[1] = 1
            e2[0] = 1;
            e2[1] = 0
        elif e1[1] == 0:
            e1[0] = 1
            e2[0] = 0
            e2[1] = 1
        else:
            norm_ = math.sqrt(e1[0]*e1[0] + e1[1]*e1[1])
            e1[0] /= norm_
            e1[1] /= norm_
            sign = math.copysign(1, e1[0]) # for whatever reason, python has no sign func
            e2[1] = abs(e1[0])
            e2[0] = -e1[1]*sign
    
    
@cuda.jit(device=True)
def get_eighen_elmts_2x2(M, l, e1, e2):
    
    get_eighen_val_2x2(M, l)
    get_eighen_vect_2x2(M, l, e1, e2)
    
    
     
@cuda.jit(device=True) 
def interpolate_cov(covs, center_pos, interpolated_cov):
    reframed_posx, _ = math.modf(center_pos[1]) # these positions are between 0 and 1
    reframed_posy, _ = math.modf(center_pos[0])
    # cov 00 is in (0,0) ; cov 01 in (0, 1) ; cov 01 in (1, 0), cov 11 in (1, 1)
    
    for i in range(2):
        for j in range(2):
            interpolated_cov[i, j] = (covs[0,0,i,j]*(1 - reframed_posx)*(1 - reframed_posy) +
                                      covs[0,1,i,j]*(reframed_posx)*(1 - reframed_posy) + 
                                      covs[1,0,i,j]*(1 - reframed_posx)*(reframed_posy) + 
                                      covs[1,1,i,j]*reframed_posx*reframed_posy)

@cuda.jit(device=True)
def bilinear_interpolation(values, pos):
    """
    

    Parameters
    ----------
    values : Array[2, 2]
        values on the 4 closest neighboors
    pos : Array[2]
        position where interpolation must be done (in [0, 1]x[0, 1]). y, x

    Returns
    -------
    val : float
        interpolated value

    """
    posy = pos[0]
    posx = pos[1]
    val = (values[0,0]*(1 - posx)*(1 - posy) +
           values[0,1]*(posx)*(1 - posy) + 
           values[1,0]*(1 - posx)*(posy) + 
           values[1,1]*posx*posy )
    return val
            
            
    
    
    
    
    
    
