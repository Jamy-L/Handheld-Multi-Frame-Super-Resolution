import torch
import torch.fft
import numpy as np
from skimage import filters

from .utils_fft import *


def gaussian_filter(sigma, theta, shift=np.array([0.0, 0.0]), k_size=np.array([15, 15])):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1, lambda_2 = sigma
    theta = -theta

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1**2, lambda_2**2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position
    MU = k_size // 2 - shift
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calculate Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ))

    # Normalize the kernel and return
    if np.sum(raw_kernel) < 1e-2:
        kernel = np.zeros_like(raw_kernel)
        kernel[k_size[0]//2, k_size[1]//2] = 1
    else:
        kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def dirac(dims):
    kernel = zeros(dims)
    hh = dims[0] // 2
    hw = dims[1] // 2
    kernel[hh, hw] = 1
    return kernel


def gaussian(images, sigma=1.0, theta=0.0):
    ## format Gaussian parameter for the gaussian_filter routine
    if isinstance(sigma, float) or isinstance(sigma, int):
        sigmas = ones(images.shape[0],2) * sigma
    elif isinstance(sigma, tuple) or isinstance(sigma, list):
        sigmas = ones(images.shape[0],2)
        sigmas[:,0] *= sigma[0]
        sigmas[:,1] *= sigma[1]
    else:
        sigmas = sigma
    if isinstance(theta, float) or isinstance(theta, int):
        thetas = ones(images.shape[0],1) * theta
    else:
        thetas = theta
    assert(theta.ndim-2)
    ## perform Gaussian filtering
    kernels = gaussian_filter(sigmas=sigmas, thetas=thetas)
    kernels = torch.to_tensor(kernels).unsqueeze(1).float().to(images.device)  # Nx1xHxW
    return conv2d(images, kernels)


def fourier_gradients(images):
    ## compute FT
    U = torch.fft.fft2(images)
    U = torch.fft.fftshift(U, dim=(-2, -1))
    ## Create the freqs components
    H, W = images.shape[-2:]
    freqh = (torch.arange(0, H, device=images.device) - H // 2)[None, None, :, None] / H
    freqw = (torch.arange(0, W, device=images.device) - W // 2)[None, None, None, :] / W
    ## Compute gradients in Fourier domain
    gxU = 2 * np.pi * freqw * (-torch.imag(U) + 1j * torch.real(U))
    gxU = torch.fft.ifftshift(gxU, dim=(-2, -1))
    gxu = torch.real(torch.fft.ifft2(gxU))
    gyU = 2 * np.pi * freqh * (-torch.imag(U) + 1j * torch.real(U))
    gyU = torch.fft.ifftshift(gyU, dim=(-2, -1))
    gyu = torch.real(torch.fft.ifft2(gyU))
    return gxu, gyu


def images_gradients(images, sigma=1.0):
    images_smoothed = fast_gaussian(images, sigma)
    gradients_x = torch.roll(images_smoothed, 1, dims=-2) - torch.roll(images_smoothed, -1, dims=-2)
    gradients_y = torch.roll(images_smoothed, 1, dims=-1) - torch.roll(images_smoothed, -1, dims=-1)
    return gradients_x, gradients_y

