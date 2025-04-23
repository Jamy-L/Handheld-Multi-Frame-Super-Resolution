import os
import glob
from pathlib import Path
import math

import cv2
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
    
    

def frame_count_denoising_gauss(image, r_acc, params):
    denoised = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    grey_mode = params['mode'] == 'grey'
    scale = params['scale']
    sigma_max = params['sigma max']
    max_frame_count = params['max frame count']
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(denoised.shape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(denoised.shape[0]/threadsperblock[0])
    blockspergrid_z = math.ceil(denoised.shape[2]/threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_frame_count_denoising_gauss[blockspergrid, threadsperblock](
        image, denoised, r_acc,
        scale, sigma_max, max_frame_count, grey_mode)
    
    return denoised

@cuda.jit
def cuda_frame_count_denoising_gauss(noisy, denoised, r_acc,
                               scale, sigma_max, max_frame_count, grey_mode):
    x, y, c = cuda.grid(3)
    imshape_y, imshape_x, _ = noisy.shape
    
    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return
    
    if grey_mode:
        y_grey = int(round(y/scale))
        x_grey = int(round(x/scale))
    else:
        y_grey = int(round((y-0.5)/(2*scale)))
        x_grey = int(round((x-0.5)/(2*scale)))
        
    r = r_acc[y_grey, x_grey]
    sigma = denoise_power_gauss(r, sigma_max, max_frame_count)
    
    t = 3*sigma
    
    num = 0
    den = 0
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            x_ = x + j
            y_ = y + i
            if (0 <= y_ < imshape_y and
                0 <= x_ < imshape_x):
                if sigma == 0:
                    w = (i==j==0)
                else:
                    w = math.exp(-(j*j + i*i)/(2*sigma*sigma))
                num += w * noisy[y_, x_, c]
                den += w
                
    denoised[y, x, c] = num/den

@cuda.jit(device=True)
def denoise_power_gauss(r_acc, sigma_max, r_max):
    r = min(r_acc, r_max)
    return sigma_max * (r_max - r)/r_max

def frame_count_denoising_median(image, r_acc, params):
    denoised = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    
    grey_mode = params['mode'] == 'grey'
    scale = params['scale']
    radius_max = params['radius max']
    max_frame_count = params['max frame count']
    
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(denoised.shape[1]/threadsperblock[1])
    blockspergrid_y = math.ceil(denoised.shape[0]/threadsperblock[0])
    blockspergrid_z = math.ceil(denoised.shape[2]/threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_frame_count_denoising_median[blockspergrid, threadsperblock](
        image, denoised, r_acc,
        scale, radius_max, max_frame_count, grey_mode)
    
    return denoised

@cuda.jit
def cuda_frame_count_denoising_median(noisy, denoised, r_acc,
                               scale, radius_max, max_frame_count, grey_mode):
    x, y, c = cuda.grid(3)
    imshape_y, imshape_x, _ = noisy.shape
    
    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return
    
    if grey_mode:
        y_grey = int(round(y/scale))
        x_grey = int(round(x/scale))
    else:
        y_grey = int(round((y-0.5)/(2*scale)))
        x_grey = int(round((x-0.5)/(2*scale)))
        
    r = r_acc[y_grey, x_grey]
    radius = denoise_power_median(r, radius_max, max_frame_count)
    radius = min(14, radius) # for memory purpose
    
    
    buffer = cuda.local.array(16*16, DEFAULT_CUDA_FLOAT_TYPE)
    k = 0
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            x_ = x + j
            y_ = y + i
            if (0 <= y_ < imshape_y and
                0 <= x_ < imshape_x):
                buffer[k] = noisy[y_, x_, c]
                k += 1
    
    bubble_sort(buffer[:k])
    
    
    denoised[y, x, c] = buffer[k//2]
    

@cuda.jit(device=True)
def denoise_power_median(r_acc, radius_max, max_frame_count):
    r = min(r_acc, max_frame_count)
    return round(radius_max * (max_frame_count - r)/max_frame_count)

@cuda.jit(device=True)
def bubble_sort(X):
    N = X.size

    for i in range(N-1):
        for j in range(N-i-1):
            if X[j] > X[j+1]:
                X[j], X[j+1] = X[j+1], X[j]
                
@cuda.jit(device=True)
def denoise_power_merge(r_acc, power_max, max_frame_count):
    if r_acc <= max_frame_count:
        return power_max
    else:
        return 1
    
@cuda.jit(device=True)
def denoise_range_merge(r_acc, rad_max, max_frame_count):
    rad_min = 1 # 3 by 3 window
    
    if r_acc <= max_frame_count:
        return rad_max
    else:
        return rad_min
    

            
                
    
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

    # Filter the image before downsampling it
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
        

        # 2 times gaussian 1d is faster than gaussian 2d
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
    
def load_png_burst(burst_path, n_images):
    """
    Loads a png burst into numpy arrays, and their exif tags.

    Parameters
    ----------
    burst_path : Path or str
        Path of the folder containing the .dngs

    Returns
    -------
    ref_raw : numpy Array[H, W]
        Reference frame
    raw_comp : numpy Array[n, H, W]
        Stack of non-reference frame
    tags : dict
        Tags of the reference frame
    reference_path
        Path of the reference image.

    """
    ref_id = 0
    raw_comp = []

    # This ensures that burst_path is a Path object
    burst_path = Path(burst_path)


    #### Read dng as numpy arrays
    # Get the list of raw images in the burst path
    raw_path_list = glob.glob(os.path.join(burst_path.as_posix(), '*.png'))
    assert len(raw_path_list) != 0, 'At least one .png file must be present in the burst folder.'
    raw_path_list = sorted(raw_path_list)
    for index, img_path in enumerate(raw_path_list):
        if index == ref_id:
            continue
        if index >= n_images:
            break
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to keep original bit depth/channels
        if img is None:
            raise ValueError(f"Could not read image at path: {img_path}")
        raw_comp.append(img)
    raw_comp = np.array(raw_comp)

    # Reference image selection and metadata
    ref_raw = cv2.imread(raw_path_list[ref_id], cv2.IMREAD_UNCHANGED)

    #### Performing whitebalance and normalizing into 0, 1
    assert ref_raw.dtype == np.uint8, NotImplementedError('Only uint8 png files are supported for now.')

    if np.issubdtype(type(ref_raw[0, 0]), np.integer):
        ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)

        ref_raw = ref_raw / 255.0
        raw_comp = raw_comp / 255.0

    return ref_raw, raw_comp, raw_path_list
