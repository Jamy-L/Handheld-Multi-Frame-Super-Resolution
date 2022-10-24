# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:41:58 2022

@author: jamyl
"""

from math import sqrt, isnan, isinf, copysign, modf

from numba import uint8, uint16, float32, float64, jit, njit, cuda

from .utils import DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_NUMPY_FLOAT_TYPE

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
def solve_6x6_krylov(A, B, X, n_iter):
    """
    Cuda function for resolving the 6x6 system A*X = B
    by using the conjuguate gradient method in krylov space.
    This function is meant to be called by a thread pool of at least 7x7 threads

    Parameters
    ----------
    A : Array[6,6]
        Definite array
    B : Array[6]
        
    X : Array[6]
        initialisation of solution 
    n_iter : int
        number of iterations. 6 is necessary to get the right solution in theory,
        but you may want to take a bit more to compensate numerical inaccuracies

    Returns
    -------
    Success : Bool
        True if the array has been succesfully inverted. Else False

    """
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    r = cuda.shared.array(6, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    d = cuda.shared.array(6, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    delta = cuda.shared.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE) # delta, delta'
    if i <= 5 and j == 0: # multithreaded init
        r[i] = B[i]
        d[i] = B[i]
    if i == 1 and j == 1:
        delta[0] = 0
    
    # waiting init
    cuda.syncthreads()
    # r = b-AX
    if i <= 5 and j <= 5:
        error = A[i, j]*X[j]
        cuda.atomic.sub(r, i, error)
        cuda.atomic.sub(d, i, error)
        
    cuda.syncthreads()
    if i<= 5 and j == 0:
        cuda.atomic.add(delta, 0, r[i]**2)
    
    cuda.syncthreads() # finishing operation

    for k in range(n_iter):
        # computing delta'
        if i ==0 and j ==0:
            delta[1] = 0
        cuda.syncthreads()
        # computing d^T A d
        if i <= 5 and j<= 5:
            cuda.atomic.add(delta, 1, A[i, j]*d[i]*d[j])
        cuda.syncthreads()
        # updating x
        lambda_ = delta[0]/delta[1]
        if i <= 5 and j == 6:
            X[i] = X[i] + lambda_*d[i]
        # updating r 
        if i <= 5 and j <= 5  :
            cuda.atomic.sub(r, i, lambda_*A[i, j]*d[j]) 
        # initing new delta
        delta_new = cuda.shared.array(1, DEFAULT_CUDA_FLOAT_TYPE)
        if i == 6 and j == 6:
            delta_new[0] = 0
        # comuting r norm
        cuda.syncthreads()
        if i <= 5 and j == 0 :
            cuda.atomic.add(delta_new, 0 ,r[i]**2)
        cuda.syncthreads()
        if isnan(delta_new[0]/delta[0]): # if A is Definite, then this is impossible. Since A is positive, this case can only happen is A is not invertible
            return False
        # updating d
        beta = (delta_new[0]/delta[0])
        if i<= 5 and j == 6 :
            d[i] = r[i] + beta*d[i]
        # updating delta
        if i == 0 and j == 0 : 
            delta[0] = delta_new[0]
    return True
    

@cuda.jit(device=True)
def solve_6x6_jacobi(A, B, X):
    """
    Cuda function for resolving the 6x6 system A*X = B
    by using the Jacobi method. This function is meant to be called by a thread
    pool of more than 6 threads

    Parameters
    ----------
    A : Array[6,6]
        diagonal dominant Array
    B : Array[6]

    Returns
    -------
    None

    """
    buffer = cuda.shared.array(6, dtype= DEFAULT_CUDA_FLOAT_TYPE)
    n_iter = 10
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    for k in range(n_iter):
        if i<= 5 and j == 0:
            buffer[i] = 0 
        cuda.syncthreads()
        if i<= 5 and j <= 5 and j != i:
            cuda.atomic.add(buffer, i, A[i, j]*X[j])
        cuda.syncthreads()
        if i<= 5 and j == 0:
            X[i] = (B[i] - buffer[i])/A[i, i]
    
    

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
    if isinf(det_i):
        M_i[0,0] = 1
        M_i[0, 1] = 0
        M_i[1, 0] = 0
        M_i [1, 1] = 1
    else:
        M_i[0,0] = M[1,1]*det_i
        M_i[0, 1] = -M[0, 1]*det_i
        M_i[1, 0] = -M[1, 0]*det_i
        M_i [1, 1] = M[0, 0]*det_i
    
@cuda.jit(device=True)
def quad_mat_prod(A, X):
    y = A[0, 0]*X[0]*X[0] + X[0]*X[1]*(A[0, 1] + A[1, 0]) + A[1, 1]*X[1]*X[1]
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
        return 0/0

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
    if M[0, 1] == 0 and M[1, 0] == 0 and M[0,0] == M[1, 1]:
        # M is multiple of identity, picking 2 ortogonal eighen vectors.
        e1[0] = 1; e1[1] = 0
        e2[0] = 0; e2[0] = 1
        
    else:
        # averaging 2 for increased reliability
        e1[0] = M[0, 0] - l[1] + M[0, 1]; e1[1] = M[1,0] + M[1,1] - l[1]
        
        if e1[0] == 0:
            e1[1] = 1
            e2[0] = 1; e2[1] = 0
        elif e1[1] == 0:
            e1[0] = 1
            e2[0] = 0; e2[1] = 1
        else:
            norm1 = sqrt(e1[0]**2 + e1[1]**2)
            e1[0] /= norm1; e1[1] /= norm1
            sign = copysign(1, e1[0]) # for whatever reason, python has no sign func
            e2[1] = abs(e1[0])
            e2[0] = -e1[1]*sign
    
    
@cuda.jit(device=True)
def get_eighen_elmts_2x2(M, l, e1, e2):
    
    get_eighen_val_2x2(M, l)
    get_eighen_vect_2x2(M, l, e1, e2)
    
    
    
    
    
@cuda.jit(device=True) # TODo this can be parallelized ! 
def interpolate_cov(covs, center_pos, interpolated_cov):
    reframed_posx, _ = modf(center_pos[1]) # these positions are between 0 and 1
    reframed_posy, _ = modf(center_pos[0])
    # cov 00 is in (0,0) ; cov 01 in (0, 1) ; cov 01 in (1, 0), cov 11 in (1, 1)
    
    for i in range(2):
        for j in range(2):
            interpolated_cov[i, j] = (covs[0,0,i,j]*(1 - reframed_posx)*(1 - reframed_posy) +
                                      covs[0,1,i,j]*(reframed_posx)*(1 - reframed_posy) + 
                                      covs[1,0,i,j]*(1 - reframed_posx)*(reframed_posy) + 
                                      covs[1,1,i,j]*reframed_posx*reframed_posy )

@cuda.jit(device=True)
def bicubic_interpolation(values, pos):
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
            
            
    
    
    
    
    
    
