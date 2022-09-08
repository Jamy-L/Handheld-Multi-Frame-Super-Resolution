import numpy as np
import torch
import torch.fft
import torch.nn.functional as F

from kornia import morphology

from . import utils_fft
from .filters import fourier_gradients

from . import edgetaper


#########################################
############# Deblurring ################
#########################################


def mild_inverse_rank3(img, kernel, alpha=2, b=3, correlate=False, halo_removal=False):
    ## (Optional) Perform correlation instead of convolution
    if correlate:
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
    ## Go to Fourier domain
    ks = kernel.shape[-1] // 2
    padding = (ks, ks, ks, ks)
    img = F.pad(img, padding, 'replicate')
    # img = edgetaper.edgetaper(img, kernel, n_tapers=3)
    Y = torch.fft.fft2(img, dim=(-2, -1))
    K = utils_fft.p2o(kernel, img.shape[-2:])  # from NxCxhxw to NxCxHxW
    ## Deblurring
    C = torch.conj(K) / (torch.abs(K) + 1e-8)
    a3 = alpha / 2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    Y = C * Y
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    imout = torch.real(torch.fft.ifft2(X, dim=(-2, -1)))
    ## (Optional) Removing deblurring halos
    if halo_removal:
        imout = remove_halos(img, imout)
    return imout[..., ks:-ks, ks:-ks]


def remove_halos(imblur, imrestored):
    grad_x, grad_y = fourier_gradients(imblur)
    gout_x, gout_y = fourier_gradients(imrestored)
    M = (-grad_x) * gout_x + (-grad_y) * gout_y
    nM = torch.sum(torch.abs(grad_x) ** 2 + torch.abs(grad_y) ** 2, dim=(-2, -1))
    z = torch.maximum(M / (nM + M), torch.zeros_like(M))
    imrestored = z * img + (1 - z) * imrestored
    return imrestored


#########################################
########## Domain transform #############
#########################################


def recursive_filter(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    I = img.clone()

    if joint_image is None:
        J = I
    else:
        J = joint_image

    batch, num_joint_channels, h, w = J.shape

    ## Compute the domain transform
    # Estimate horizontal and vertical partial derivatives using finite differences
    dIcdx = torch.diff(J, n=1, dim=-1)
    dIcdy = torch.diff(J, n=1, dim=-2)

    # compute the l1-norm distance of neighbor pixels.
    dIdx = torch.zeros(batch, h, w, device=I.device)
    dIdx[:, :, 1:] = torch.sum(torch.abs(dIcdx), dim=1)
    dIdy = torch.zeros(batch, h, w, device=I.device)
    dIdy[:, 1:, :] = torch.sum(torch.abs(dIcdy), dim=1)

    # compute the derivatives of the horizontal and vertical domain transforms
    dHdx = (1 + sigma_s/sigma_r * dIdx)
    dVdy = (1 + sigma_s/sigma_r * dIdy)

    # the vertical pass is performed using a transposed image
    dVdy = dVdy.transpose(-2, -1)

    ## Perform the filtering
    N = num_iterations
    F = I

    sigma_H = sigma_s

    for i in range(num_iterations):
        # Compute the sigma value for this iterations (Equation 14 of our paper)
        sigma_H_i = sigma_H * np.sqrt(3) * 2**(N - (i + 1)) / np.sqrt(4**N - 1)

        F = transformed_domain_recursive_filter_horizontal(F, dHdx, sigma_H_i)
        F = F.transpose(-1, -2)

        F = transformed_domain_recursive_filter_horizontal(F, dVdy, sigma_H_i)
        F = F.transpose(-1, -2)

    F = F.type_as(img)

    return F


def transformed_domain_recursive_filter_horizontal(I, D, sigma):
    # Feedback coefficient (Appendix of our paper).
    a = np.exp(-np.sqrt(2) / sigma)

    F = I
    V = a**D

    batch, num_channels, h, w = I.shape
    # batch, h, w = D.shape

    # Left -> Right filter
    for i in range(1, w, 1):
        for c in range(num_channels):
            F[:, c, :, i] = F[:, c, :, i] + V[:, :, i] * (F[:, c, :, i - 1] - F[:, c, :, i])

    # Right -> Left filter
    for i in range(w-2, -1, -1):  # from w-2 to 0
        for c in range(num_channels):
            F[:, c, :, i] = F[:, c, :, i] + V[:, :, i + 1] * (F[:, c, :, i + 1] - F[:, c, :, i])

    return F


#########################################
########## Blur estimation ##############
#########################################


def blur_estimation(img, sigma_b, c, ker_size, q=0.0001, n_angles=6, n_interpolated_angles=30):
    # flag saturated areas
    mask = compute_mask(img)

    # normalized images
    img_normalized = normalize(img, q=q)

    # compute the image gradients
    gradients = compute_gradients(img_normalized)

    # compute the gradiennt magnitudes per orientation
    gradients_magnitude = compute_gradient_magnitudes(gradients, n_angles=n_angles)

    # find the maximal direction amongst sampled orientations
    magnitude_normal, magnitude_ortho, theta = find_blur_direction(gradients, gradients_magnitude, mask, n_angles=n_angles,
                                                                   n_interpolated_angles=n_interpolated_angles)

    # compute the Gaussian parameters
    sigma, rho = compute_gaussian_parameters(magnitude_normal, magnitude_ortho, c=c, sigma_b=sigma_b)

    # create the blur kernel
    kernel = create_gaussian_filter(theta, sigma, rho, ksize=ker_size, device=img.device)

    return kernel


def compute_mask(img, crop=11):
    mask = img > 0.99
    mask = morphology.dilation(mask.float(), torch.ones(5,5).to(mask.device), border_type='replicate', engine='convolution')
    mask = mask.bool()
    mask[..., :crop, :] = 0
    mask[..., -crop:, :] = 0
    mask[..., :, :crop] = 0
    mask[..., :, -crop:] = 0
    return mask


def normalize(img, q):
    b, c, h, w = img.shape
    value_min = torch.quantile(img.reshape(b, c, -1), q=q, dim=-1, keepdim=True).unsqueeze(-1)  # (b,c,1,1)
    value_max = torch.quantile(img.reshape(b, c, -1), q=1-q, dim=-1, keepdims=True).unsqueeze(-1)  # (b,c,1,1)
    img = (img - value_min) / (value_max - value_min)
    return img.clamp(0.0, 1.0)


def compute_gradients(img, mask=None):
    gradient_x, gradient_y = fourier_gradients(img)
    if mask is not None:
        gradient_x[mask] = 0
        gradient_y[mask] = 0
    return gradient_x, gradient_y


def compute_gradient_magnitudes(gradients, n_angles=6):
    gradient_x, gradient_y = gradients  # (B,C,H,W)
    gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    angles = torch.linspace(0, np.pi, n_angles + 1, device=gradient_x.device).view(1, -1, 1, 1, 1)  # (1,N,1,1,1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    gradient_magnitudes_angles = (cos * gradient_x_gray - sin * gradient_y_gray).abs()  # (B,N,1,H,W)
    gradient_magnitudes_angles = torch.amax(gradient_magnitudes_angles, dim=(-3, -2, -1))  # (B,N)
    return gradient_magnitudes_angles


def cubic_interpolator(x_new, x, y):
    """
    Key's convolutional approximation of cubic interpolation
    """
    abs_s = torch.abs(x_new[..., None] - x[..., None, :])
    u = torch.zeros_like(abs_s)
    # 0 < |s| < 1
    mask = abs_s < 1
    u[mask] = 1
    abs_s_mask = abs_s[mask]
    abs_s_pow = abs_s_mask * abs_s_mask  # ^2
    u[mask] += -2.5 * abs_s_pow
    abs_s_pow *= abs_s_mask  # ^3
    u[mask] += 1.5 * abs_s_pow
    # 1 <= |s| < 2
    mask = torch.bitwise_and(1 <= abs_s, abs_s < 2)
    u[mask] = 2
    abs_s_mask = abs_s[mask]
    abs_s_pow = abs_s_mask  # ^1
    u[mask] += -4 * abs_s_pow
    abs_s_pow *= abs_s_mask  # ^2
    u[mask] += 2.5 * abs_s_pow
    abs_s_pow *= abs_s_mask  # ^3
    u[mask] += -0.5 * abs_s_pow
    u = u / (torch.sum(u, dim=-1, keepdim=True) + 1e-8)
    y_new = (u @ y[..., None]).squeeze(-1)
    return y_new


def find_blur_direction(gradients, gradient_magnitudes_angles, mask, n_angles=6, n_interpolated_angles=30):
    gradient_x, gradient_y = gradients
    b = gradient_x.shape[0]
    ## Find thetas
    thetas = torch.linspace(0, 180, n_angles+1, device=gradient_magnitudes_angles.device).unsqueeze(0)  # (1,n)
    interpolated_thetas = torch.arange(0, 180, 180 / n_interpolated_angles, device=thetas.device).unsqueeze(0)  # (1,N)
    gradient_magnitudes_interpolated_angles = cubic_interpolator(interpolated_thetas / n_interpolated_angles,
                                                thetas / n_interpolated_angles, gradient_magnitudes_angles)  # (B,N)
    ## Compute magnitude in theta
    i_min = torch.argmin(gradient_magnitudes_interpolated_angles, dim=-1, keepdim=True).long()
    thetas_normal = torch.take_along_dim(interpolated_thetas, i_min, dim=-1)
    gradient_color_magnitude_normal = gradient_x * torch.cos(thetas_normal.view(-1, 1, 1, 1) * np.pi / 180) - \
                                      gradient_y * torch.sin(thetas_normal.view(-1, 1, 1, 1) * np.pi / 180)
    magnitudes_normal = torch.abs(gradient_color_magnitude_normal)
    magnitudes_normal[mask] = 0
    magnitudes_normal, _ = torch.max(magnitudes_normal.view(b, 3, -1), dim=-1)
    ## Compute magnitude in theta+90
    thetas_ortho = (thetas_normal + 90.0) % 180  # angle in [0,pi)
    gradient_color_magnitude_ortho = gradient_x * torch.cos(thetas_ortho.view(-1, 1, 1, 1) * np.pi / 180) - \
                                     gradient_y * torch.sin(thetas_ortho.view(-1, 1, 1, 1) * np.pi / 180)
    magnitudes_ortho = torch.abs(gradient_color_magnitude_ortho)
    magnitudes_ortho[mask] = 0
    magnitudes_ortho, _ = torch.max(magnitudes_ortho.view(b, 3, -1), dim=-1)
    return magnitudes_normal, magnitudes_ortho, thetas_normal * np.pi / 180


def compute_gaussian_parameters(magnitudes_normal, magnitudes_ortho, c, sigma_b):
    ## Compute sigma
    sigma = c**2 / (magnitudes_normal ** 2 + 1e-8) - sigma_b**2
    sigma = torch.maximum(sigma, 0.09 * torch.ones_like(sigma))
    sigma[sigma > 16] = 0.09
    sigma = torch.sqrt(sigma)
    sigma[magnitudes_normal < 0.1] = 0.3
    ## Compute rho
    rho = c**2 / (magnitudes_ortho ** 2 + 1e-8) - sigma_b**2
    rho = torch.maximum(rho, 0.09 * torch.ones_like(rho))
    rho[rho > 16] = 0.09
    rho = torch.sqrt(rho)
    rho[magnitudes_ortho < 0.1] = 0.3
    return sigma, rho


def create_gaussian_filter(thetas, sigmas, rhos, ksize, device):
    B = len(sigmas)
    C = 3
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = sigmas
    lambda_2 = rhos
    # thetas = -thetas

    # Set COV matrix using Lambdas and Theta
    LAMBDA = torch.zeros(B, C, 2, 2) # (B,C,2,2)
    LAMBDA[:, :, 0, 0] = lambda_1**2
    LAMBDA[:, :, 1, 1] = lambda_2**2
    Q = torch.zeros(B, C, 2, 2)  # (B,C,2,2)
    Q[:, :, 0, 0] = torch.cos(thetas)
    Q[:, :, 0, 1] = -torch.sin(thetas)
    Q[:, :, 1, 0] = torch.sin(thetas)
    Q[:, :, 1, 1] = torch.cos(thetas)
    SIGMA = torch.einsum("bcij,bcjk,bckl->bcil", [Q, LAMBDA, Q.transpose(-2, -1)])  # (B,C,2,2)
    INV_SIGMA = torch.linalg.inv(SIGMA)
    INV_SIGMA = INV_SIGMA.view(B, C, 1, 1, 2, 2)  # (B,C,1,1,2,2)

    # Set expectation position
    MU = (ksize//2) * torch.ones(B, C, 2)
    MU = MU.view(B, C, 1, 1, 2, 1)  # (B,C,1,1,2,1)

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(ksize),
                          torch.arange(ksize),
                          indexing='xy')
    Z = torch.stack([X, Y], dim=-1).unsqueeze(-1)  # (k,k,2,1)

    # Calculate Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(-2, -1)  # (B,C,k,k,1,2)
    raw_kernels = torch.exp(-0.5 * (ZZ_t @ INV_SIGMA @ ZZ).squeeze(-1).squeeze(-1))  # (B,C,k,k)

    # Normalize the kernel and return
    mask_small = torch.sum(raw_kernels, dim=(-2, -1)) < 1e-2
    if mask_small.any():
        raw_kernels[mask_small].copy_(0)
        raw_kernels[mask_small, ksize//2, ksize//2].copy_(1)
    kernels = raw_kernels / torch.sum(raw_kernels, dim=(-2, -1), keepdim=True)
    return kernels.to(device)
