import math

import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d
from numba import cuda
import torch as th
import torch.fft
import torch.nn.functional as F

from .utils import getSigned, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS

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

def compute_grey_images(img, method):
    """
    This function converts a raw image to a grey image, using the decimation or
    the method of Alg. 3: ComputeGrayscaleImage

    Parameters
    ----------
    img : device Array[:, :]
        Raw image J to convert to gray level.
    method : str
        FFT or decimatin.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    img_grey : device Array[:, :]
        Corresponding grey scale image G

    """
    imsize_y, imsize_x = img.shape
    if method == "FFT":
        torch_img_grey = th.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
        torch_img_grey = torch.fft.fft2(torch_img_grey) 
        # th FFT induces copy on the fly : this is good because we dont want to 
        # modify the raw image, it is needed in the future
        # Note : the complex dtype of the fft2 is inherited from DEFAULT_TORCH_FLOAT_TYPE.
        # Therefore, for DEFAULT_TORCH_FLOAT_TYPE = float32 we directly get complex64
        torch_img_grey = torch.fft.fftshift(torch_img_grey)
        
        torch_img_grey[:imsize_y//4, :] = 0
        torch_img_grey[:, :imsize_x//4] = 0
        torch_img_grey[-imsize_y//4:, :] = 0
        torch_img_grey[:, -imsize_x//4:] = 0
        
        torch_img_grey = torch.fft.ifftshift(torch_img_grey)
        torch_img_grey = torch.fft.ifft2(torch_img_grey)
        # Here, .real() type inherits once again from the complex type.
        # numba type is read directly from the torch tensor, so everything goes fine.
        return cuda.as_cuda_array(torch_img_grey.real)
    elif method == "decimating":
        grey_imshape_y, grey_imshape_x = grey_imshape = imsize_y//2, imsize_x//2
        
        img_grey = cuda.device_array(grey_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
        
        threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
        blockspergrid_x = math.ceil(grey_imshape_x/threadsperblock[1])
        blockspergrid_y = math.ceil(grey_imshape_y/threadsperblock[0])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        cuda_decimate_to_grey[blockspergrid, threadsperblock](img, img_grey)
        return img_grey
        
    else:
        raise NotImplementedError('Computation of gray level on GPU is only supported for FFT')

def GAT(image, alpha, beta):
    """
    Generalized Ascombe Transform
    noise model : std² = alpha * I + beta
    Where alpha and beta are iso dependant. 

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    alpha : float
        value of alpha for the given iso 
    iso : float
        ISO value
    beta : float
        Value of beta for the given iso

    Returns
    -------
    VST_image : TYPE
        input image with stabilized variance

    """
    assert len(image.shape) == 2
    assert alpha > 0, f"alpha should be positive, got {alpha} (VST is ill defined and kernels would be wrong)"
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
    
    # ISO should not appear here,  since alpha and beta are
    # already iso dependant.
    VST = alpha*image[y, x] + 3/8 * alpha*alpha + beta
    VST = max(0, VST)
    
    VST_image[y, x] = 2/alpha * math.sqrt(VST)     
                
    
def fft_lowpass(img_grey):
    img_grey = th.from_numpy(img_grey).to("cuda")
    img_grey = torch.fft.fft2(img_grey)
    img_grey = torch.fft.fftshift(img_grey)
    
    imsize_y, imsize_x = img_grey.shape
    img_grey[:imsize_y//4, :] = 0
    img_grey[:, :imsize_x//4] = 0
    img_grey[-imsize_y//4:, :] = 0
    img_grey[:, -imsize_x//4:] = 0
    
    img_grey = torch.fft.ifftshift(img_grey)
    img_grey = torch.fft.ifft2(img_grey)
    return img_grey.cpu().numpy().real

@cuda.jit
def cuda_decimate_to_grey(img, grey_img):
    x, y = cuda.grid(2)
    grey_imshape_y, grey_imshape_x = grey_img.shape
    
    if (0 <= y < grey_imshape_y and
        0 <= x < grey_imshape_x):
        c = 0
        for i in range(0, 2):
            for j in range(0, 2):
                c += img[2*y + i, 2*x + j]
        grey_img[y, x] = c/4
        

def cuda_downsample(th_img, kernel='gaussian', factor=2):
    '''Apply a convolution by a kernel if required, then downsample an image.
    Args:
     	image: Device Array the input image (WARNING: single channel only!)
     	kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
     	factor: downsampling factor
    '''
    # Special case
    if factor == 1:
        return th_img
    
    if kernel is None:
        raise ValueError('use Kernel')
    elif kernel == 'gaussian':
        # gaussian kernel std is proportional to downsampling factor
        # filteredImage = gaussian_filter(image, sigma=factor * 0.5, order=0, output=None, mode='reflect')
        
        # This is the default kernel of scipy gaussian_filter1d
        # Note that pytorch Convolve is actually a correlation, hence the ::-1 flip.
        # copy to avoid negative stride
        gaussian_kernel = _gaussian_kernel1d(sigma=factor * 0.5, order=0, radius=int(4*factor * 0.5 + 0.5))[::-1].copy()
        th_gaussian_kernel = torch.as_tensor(gaussian_kernel, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")

        temp = F.conv2d(th_img, th_gaussian_kernel[None, None, :, None]) # convolve y
        th_filteredImage = F.conv2d(temp, th_gaussian_kernel[None, None, None, :]) # convolve x
    else:
        raise ValueError("please use gaussian kernel")

    # Shape of the downsampled image
    h2, w2 = np.floor(np.array(th_filteredImage.shape[2:]) / float(factor)).astype(int)

    return th_filteredImage[:, :, :h2 * factor:factor, :w2 * factor:factor]


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
        print('WARNING: images have different sizes: {}, {}. Returning None'.format(image.shape, noisyImage.shape))
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