import math

import numpy as np
import torch as th
import torch.fft
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy.typing import NDArray
from scipy.ndimage import convolve1d

from .config import GrayscaleMethod
from .utils import (
    DEFAULT_NUMPY_FLOAT_TYPE,
    DEFAULT_THREADS,
    DEFAULT_TORCH_FLOAT_TYPE,
    getSigned,
)


def apply_orientation(img, ori):
    """
    Applies an orientation to an image

    Parameters
    ----------
    img : numpy Array [ny, nx, c]
        Image
    ori : int
        Exif orientation as defined here:
            https://exiftool.org/TagNames/EXIF.html

    Returns
    -------
    Oriented image

    """
    
    if ori == 1:
        pass
    elif ori == 2:
        # Mirrored horizontal
        img = np.flip(img, axis=1)
    elif ori == 3:
        # Rotate 180
        img = np.rot90(img, k=2, axes=(0, 1))
    elif ori == 4:
        # Mirror vertical
        img = np.flip(img, axis=0)
    elif ori == 5:
        # Mirror horizontal and rotate 270 CW
        img = np.flip(img, axis=1)
        img = np.rot90(img, k=-3, axes=(0, 1))
    elif ori == 6:
        # Rotate 90 CW
        img = np.rot90(img, k=-1, axes=(0, 1))
    elif ori == 7:
        # Mirror horizontal and rotate 90 CW
        img = np.flip(img, axis=1)
        img = np.rot90(img, k=-1, axes=(0, 1))
    elif ori == 8:
        # Rotate 270 CW
        img = np.rot90(img, k=-3, axes=(0, 1))
    
    return img

def compute_grey_images(image: NDArray, method: GrayscaleMethod) -> NDArray[np.float32]:
    """Convert normalized RGGB Bayer data to NumPy gray for CPU alignment."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or any(size % 2 for size in image.shape):
        raise ValueError("Bayer input must be a 2D image with even dimensions")
    if method is GrayscaleMethod.DECIMATING:
        gray = (image[::2, ::2] + image[::2, 1::2]
                + image[1::2, ::2] + image[1::2, 1::2]) * 0.25
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        gray = convolve1d(convolve1d(gray, kernel, axis=0, mode="nearest"),
                          kernel, axis=1, mode="nearest")
    elif method is GrayscaleMethod.DEMOSAICING:
        gray = demosaic_to_grey(image)
    else:
        raise ValueError(f"Unknown grayscale method: {method}")
    # Demosaicing can overshoot the normalized sensor range; the native API
    # requires finite grayscale samples in [0, 1].
    return np.ascontiguousarray(np.clip(gray, 0, 1), dtype=np.float32)

def average_bayer_quads(img: DeviceNDArray) -> DeviceNDArray:
    """Average Bayer quads on the GPU for merging kernel estimation."""
    image = th.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    gray = (image[::2, ::2] + image[::2, 1::2]
            + image[1::2, ::2] + image[1::2, 1::2]) * 0.25
    return cuda.as_cuda_array(gray)


def demosaic_to_grey(img: NDArray) -> NDArray[np.float32]:
    """Demosaic a normalized RGGB CFA and collapse its RGB channels to gray."""
    if img.ndim != 2:
        raise ValueError(f"Bayer input must be 2D, got shape {img.shape}")
    try:
        from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    except ImportError as error:
        raise ImportError(
            "The demosaicing alignment method requires colour-demosaicing"
        ) from error

    rgb = demosaicing_CFA_Bayer_Menon2007(img, pattern="RGGB")
    gray = 0.25*(rgb[:, :, 0] + 2*rgb[:, :, 1] + rgb[:, :, 2]) # Luma conversion, consistent with quad averaging
    return np.ascontiguousarray(gray, dtype=DEFAULT_NUMPY_FLOAT_TYPE)

def preprocess(img: torch.Tensor, preprocessing: str):
    if preprocessing == "sqrt":
        return img.clamp(min=0).sqrt()
    elif preprocessing == "gamma":
        return img.clamp(min=0).pow(1/2.2)
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")

def gat(image: DeviceNDArray, alpha: tuple[float, float, float, float], beta: tuple[float, float, float, float]) -> DeviceNDArray:
    """
    Generalized Ascombe Transform
    noise model : std² = alpha * I + beta

    The input is supposed to be a mosaiced raw image
    r g1
    g2 b

    and alpha, beta given as [r g1 b g2]    



    """
    assert len(image.shape) == 2, f"image should be 2D, got {image.shape}"
    assert len(alpha) == 4, f"alpha should be of length 4, got {len(alpha)}"
    assert len(beta) == 4, f"beta should be of length 4, got {len(beta)}"
    assert all(a > 0 for a in alpha), f"alpha should be positive, got {alpha} (VST is ill defined and kernels would be wrong)"
    imshape_y, imshape_x = image.shape
    
    VST_image = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(imshape_x/threadsperblock[1])
    blockspergrid_y = math.ceil(imshape_y/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cuda_GAT[blockspergrid, threadsperblock](image, VST_image,
                                             alpha, beta)
    
    return VST_image

@cuda.jit
def cuda_GAT(image, VST_image, alpha, beta):
    x, y = cuda.grid(2)
    imshape_y,  imshape_x = image.shape
    
    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return

    # The mosaic is spatial R G1 / G2 B, whereas EXIF alpha and beta use
    # color-plane order R, G1, B, G2.
    # x even y even -> r -> alpha[0], beta[0]
    # x odd y even -> g1 -> alpha[1], beta[1]
    # x even y odd -> g2 -> alpha[3], beta[3]
    # x odd y odd -> b -> alpha[2], beta[2]
    px = x & 1
    py = y & 1
    # Spatial R,G1/G2,B -> EXIF plane indices 0,1/3,2, without divergence.
    i = px + py * (3 - 2 * px)
    alpha_ = alpha[i]
    beta_ = beta[i]
    VST = alpha_*image[y, x] + 3/8 * alpha_*alpha_ + beta_
    VST = max(0, VST)
    
    VST_image[y, x] = 2/alpha_ * math.sqrt(VST)     
                
    
@cuda.jit(device=True)
def dogson_biquadratic_kernel(x, y):
    return dogson_quadratic_kernel(x) * dogson_quadratic_kernel(y)

@cuda.jit(device=True)
def dogson_quadratic_kernel(x):
    abs_x = abs(x)
    if abs_x <= 0.5:
        return -2 * abs_x*abs_x +1
    elif abs_x <= 1.5:
        return abs_x*abs_x - 5/2 * abs_x + 1.5
    else:
        return 0

def computeRMSE(image1, image2):
    '''computes the Root Mean Square Error between two images'''
    assert np.array_equal(image1.shape, image2.shape), 'images have different sizes'
    h, w = image1.shape[:2]
    c = 1
    if len(image1.shape) == 3:  # multi-channel image
        c = image1.shape[-1]
    error = getSigned(image1.reshape(h * w * c)) - getSigned(image2.reshape(h * w * c))
    return np.sqrt(np.mean(np.multiply(error, error)))


def computePSNR(image, noisyImage):
    '''computes the Peak Signal-to-Noise Ratio between a "clean" and a "noisy" image'''
    if np.array_equal(image.shape, noisyImage.shape):
        assert image.dtype == noisyImage.dtype, 'images have different data types'
        if np.issubdtype(image.dtype, np.unsignedinteger):
            maxValue = np.iinfo(image.dtype).max
        else:
            assert(np.issubdtype(image.dtype, np.floating) and np.min(image) >= 0. and np.max(image) <= 1.), 'not a float image between 0 and 1'
            maxValue = 1.
        h, w = image.shape[:2]
        c = 1
        if len(image.shape) == 3:  # multi-channel image
            c = image.shape[-1]
        error = np.abs(getSigned(image.reshape(h * w * c)) - getSigned(noisyImage.reshape(h * w * c)))
        mse = np.mean(np.multiply(error, error))
        return 10 * np.log10(maxValue**2 / mse)
    else:
        print(f'WARNING: images have different sizes: {image.shape}, {noisyImage.shape}. Returning None')
        return None

def cfa_to_rggb(x: np.ndarray, source_cfa: np.ndarray):
    assert x.ndim in (2, 3), f"Expected 2 or 3 dim, got {x.shape}"
    assert source_cfa.shape == (2, 2), f"expected cfa of shape (2, 2), got {source_cfa.shape}"

    if np.array_equal(source_cfa, np.array([[0, 1], [1, 2]])):
        return x

    # BGGR
    if np.array_equal(source_cfa, np.array([[2, 1], [1, 0]])):
        return np.flip(x, axis=(-1, -2))
    
    # GBRG
    if np.array_equal(source_cfa, np.array([[1, 0], [2, 1]])):
        return np.flip(x, axis=-1)
    # GRGB
    if np.array_equal(source_cfa, np.array([[1, 2], [0, 1]])):
        return np.flip(x, axis=-2)
    
    raise NotImplementedError(f"Unsupported CFA pattern {source_cfa}")

def rggb_to_cfa(x: np.ndarray, target_cfa: np.ndarray):
    # the function is its own inverse...
    return cfa_to_rggb(x, target_cfa)

def estimate_image_snr(ref_img: NDArray[np.float32], alpha: tuple[float, float, float, float], beta: tuple[float, float, float, float]) -> float:
    r = ref_img[::2, ::2]
    g1 = ref_img[::2, 1::2]
    g2 = ref_img[1::2, ::2]
    b = ref_img[1::2, 1::2]

    varr =  alpha[0]*r + beta[0]
    varg1 =  alpha[1]*g1 + beta[1]
    varg2 =  alpha[3]*g2 + beta[3]
    varb =  alpha[2]*b + beta[2]

    validr = (r > 0) & (r < 1)
    validg1 = (g1 > 0) & (g1 < 1)
    validg2 = (g2 > 0) & (g2 < 1)
    validb = (b > 0) & (b < 1)

    top = np.concatenate((r[validr], g1[validg1], g2[validg2], b[validb]))
    bot = np.concatenate((varr[validr], varg1[validg1], varg2[validg2], varb[validb]))

    SNR = np.sqrt(np.sum(top*top)/np.sum(bot))
    SNR = 20 * np.log10(SNR)
    return SNR
