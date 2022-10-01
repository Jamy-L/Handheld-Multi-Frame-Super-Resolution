import cmath

import numpy as np
import numba as nb


DEFAULT_CUDA_FLOAT_TYPE = nb.float32
DEFAULT_CUDA_COMPLEX_TYPE = nb.complex64
DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_NUMPY_COMPLEX_TYPE = nb.complex64


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
    n = len(arr)
    levels = ilog2(n)
    e_arr = np.empty_like(arr)
    coeff = (-2j if direct else 2j) * cmath.pi / n
    for i in range(n):
        e_arr[i] = cmath.exp(coeff * i)
    result = np.empty_like(arr)
    for i in range(n):
        result[i] = arr[reverse_bits(i, levels)]
    # Radix-2 decimation-in-time FFT
    size = 2
    while size <= n:
        half_size = size // 2
        step = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + half_size):
                temp = result[j + half_size] * e_arr[k]
                result[j + half_size] = result[j] - temp
                result[j] += temp
                k += step
        size *= 2
    return result


@nb.njit(fastmath=True)
def fft_1d_arb(arr, fft_1d_r2=fft_1d_radix2_rbi):
    """1D FFT for arbitrary inputs using chirp z-transform"""
    arr = np.asarray(arr, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    n = len(arr)
    m = 1 << (ilog2(n) + 2)
    e_arr = np.empty(n, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    for i in range(n):
        e_arr[i] = cmath.exp(-1j * cmath.pi * (i*i) / n)
    result = np.zeros(m, dtype=DEFAULT_NUMPY_COMPLEX_TYPE)
    result[:n] = arr * e_arr
    coeff = np.zeros_like(result)
    coeff[:n] = e_arr.conjugate()
    coeff[-n + 1:] = e_arr[:0:-1].conjugate()
    return fft_convolve(result, coeff, fft_1d_r2)[:n] * e_arr / m


@nb.njit(fastmath=True)
def fft_convolve(a_arr, b_arr, fft_1d_r2=fft_1d_radix2_rbi):
    return fft_1d_r2(fft_1d_r2(a_arr) * fft_1d_r2(b_arr), False)


@nb.njit(fastmath=True)
def fft_1d(arr):
    """Main function"""
    n = len(arr)
    if not n & (n-1):
        return fft_1d_radix2_rbi(arr)
    else:
        return fft_1d_arb(arr)


if __name__ == '__main__':
    import time
    arr = np.arange(30)
    
    start = time.time()
    res_np = np.fft.fft(arr)
    print('numpy', res_np)
    print(time.time() - start)
    
    fft_1d(arr)
    start = time.time()
    res_nb = fft_1d(arr)
    print('numba', res_nb)
    print(time.time() - start)

    print('diff', np.linalg.norm(res_nb - res_np))
