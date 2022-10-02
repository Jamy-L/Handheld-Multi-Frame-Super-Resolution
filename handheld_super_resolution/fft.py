import cmath

import numpy as np
import numba as nb

from .utils import  DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_CUDA_COMPLEX_TYPE, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_NUMPY_COMPLEX_TYPE


# According to the colab sheet here
# https://colab.research.google.com/drive/1pKDJFKASDA4XrnXQrRgvbRc8iueZG2dZ?usp=sharing#scrollTo=o5MPxPdfL-67
# this numba-friendly code is fast enough to process small signals, e.g., 20 to 30.

@nb.jit
def ilog2(n):
    result = -1
    if n < 0:
        n = -n
    while n > 0:
        n >>= 1
        result += 1
    return result


@nb.njit(fastmath=True)
def reverse_bits(val, width):
    result = 0
    for _ in range(width):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


@nb.njit(fastmath=True)
def fft_1d_radix2_rbi(arr, direct=True):
    arr = np.asarray(arr, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    n = arr.shape[-1]
    levels = ilog2(n)

    e_arr = np.empty(n, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    coeff = (-2j if direct else 2j) * cmath.pi / n
    for i in range(n):
        e_arr[i] = cmath.exp(coeff * i)

    result = np.empty_like(arr)
    for i in range(n):
        result[..., i] = arr[..., reverse_bits(i, levels)]

    # Radix-2 decimation-in-time FFT
    size = 2
    while size <= n:
        half_size = size // 2
        step = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + half_size):
                temp = result[..., j + half_size] * e_arr[k]
                result[..., j + half_size] = result[..., j] - temp
                result[..., j] += temp
                k += step
        size *= 2
    return result


@nb.njit(fastmath=True)
def fft_1d_arb(arr, fft_1d_r2=fft_1d_radix2_rbi, direct=True):
    """1D FFT for arbitrary inputs using chirp z-transform"""
    arr = np.asarray(arr, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    n = arr.shape[-1]
    m = 1 << (ilog2(n) + 2)

    e_arr = np.empty(n, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    for i in range(n):
        e_arr[i] = cmath.exp(-1j * cmath.pi * (i*i) / n)

    result = np.zeros((*arr.shape[:-1], m), dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    result[..., :n] = arr * e_arr

    coeff = np.zeros_like(result)
    coeff[..., :n] = e_arr.conjugate()
    coeff[..., -n + 1:] = e_arr[:0:-1].conjugate()

    if direct:
        return fft_convolve(result, coeff, fft_1d_r2, not direct)[..., :n] * e_arr / m
    else:
        arr = fft_convolve(result, coeff, fft_1d_r2, not direct)[..., :n] * e_arr / m
        arr[..., 1:] = arr[..., :0:-1]
        return arr / n # do normalization in backward pass


@nb.njit(fastmath=True)
def fft_convolve(a_arr, b_arr, fft_1d_r2=fft_1d_radix2_rbi, direct=False):
    return fft_1d_r2(fft_1d_r2(a_arr, direct=not direct) * fft_1d_r2(b_arr, direct=not direct), direct)


@nb.njit(fastmath=True)
def fft(arr):
    n = arr.shape[-1]
    if not n & (n-1):
        return fft_1d_radix2_rbi(arr, direct=True)
    else:
        return fft_1d_arb(arr, direct=True)


@nb.njit(fastmath=True)
def ifft(arr, axis=-1):
    n = arr.shape[axis]
    if not n & (n-1):
        return fft_1d_radix2_rbi(arr, direct=False)
    else:
        return fft_1d_arb(arr, direct=False)


@nb.njit(fastmath=True)
def fft2(arr):
    arr = fft(arr)
    arr = np.swapaxes(arr, arr.ndim-1, arr.ndim-2)
    arr = fft(arr)
    arr = np.swapaxes(arr, arr.ndim-1, arr.ndim-2)
    return arr


@nb.njit(fastmath=True) 
def ifft2(arr):
    arr = ifft(arr)
    arr = np.swapaxes(arr, arr.ndim-1, arr.ndim-2)
    arr = ifft(arr)
    arr = np.swapaxes(arr, arr.ndim-1, arr.ndim-2)
    return arr


if __name__ == '__main__':
    import time


    # # arr = np.arange(9)
    # # arr = np.arange(9)[None,:]
    # arr = np.arange(10)
    # arr = np.stack([arr, arr], axis=0)
    # arr = arr[None]
    # print('arr.shape', arr.shape)
    # print('arr', arr)
    # 
    # start = time.time()
    # res_np = np.fft.fft(arr)
    # print('FFT time np', time.time() - start)
    # # print('numpy', res_np)
    # 
    # fft(arr)
    # start = time.time()
    # res_nb = fft(arr)
    # print('FFT time nb:', time.time() - start)
    # # print('numba', res_nb)
    # print('FFT diff', np.linalg.norm(res_nb - res_np))

    # ifft(arr)
    # start = time.time()
    # arr_nb = np.real(ifft(res_nb))
    # print('IFFT time nb', time.time() - start)
    # # print('arr_nb', arr_nb)
    # print('IFFT diff res', np.linalg.norm(arr_nb - arr))




    # arr = np.arange(9)
    arr = np.arange(24)
    arr = arr[:, None] * arr[None, :]
    arr = arr[None]
    print('arr.shape', arr.shape)
    print('arr', arr)

    start = time.time()
    res_np = np.fft.fft2(arr)
    print('FFT2 time np', time.time() - start)
    # print('numpy', res_np)
    
    fft2(arr)
    start = time.time()
    res_nb = fft2(arr)
    print('FFT2 time nb', time.time() - start)
    # print('numba', res_nb)
    print('FFT2 diff', np.linalg.norm(res_nb - res_np) / np.prod(arr.shape))

    # ifft2(res_nb)
    ifft2(res_nb.astype(np.complex64))  # ifft2 has to be compiled with np.complex64 input
    start = time.time()
    arr_nb = ifft2(res_nb).real
    print('IFFT2 time nb', time.time() - start)
    print('arr_nb', arr_nb)
    print('IFFT2 diff res', np.linalg.norm(arr_nb - arr) / np.prod(arr.shape))
