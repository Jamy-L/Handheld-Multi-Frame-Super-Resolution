"""
Created on Sat Aug  6 17:26:48 2022

This script contains the implementation of Alg. 5: ComputeKernelCovariance,
where the merge kernels are estimated


@author: jamyl
"""

import time
from pathlib import Path

import slangpy as spy
from numba import cuda

from .config import Config
from .utils import (
    getTime,
)

SEL_HARD_THRESHOLD = 0
SEL_LINEAR = 1


def estimate_precision_matrices(img: spy.Texture, config: Config):
    """
    Implementation of Alg. 5: ComputeKernelCovariance
    Returns the kernels covariance matrices for the frame J_n, sampled at the
    center of every bayer quad (or at the center of every grey pixel in grey
    mode).

    Parameters
    ----------
    img : device Array[imshape_y, imshape_x]
        Raw image J_n
    options : dict
        options
    config : Config
        parameters.

    Returns
    -------
    covs : device Array[imshape_y//2, imshape_x//2, 2, 2]
        Covariance matrices Omega_n, sampled at the center of each bayer quad.

    """
    verbose_3 = config.verbose >= 3
    t1 = 0.0

    sigma_detail = float(config.merging.kernel.sigma_detail)
    k_denoise = float(config.merging.kernel.k_denoise)
    D_th = float(config.merging.kernel.D_th)
    D_tr = float(config.merging.kernel.D_tr)
    k_stretch = float(config.merging.kernel.k_stretch)
    k_shrink = float(config.merging.kernel.k_shrink)

    alpha = config.noise_model.alpha
    beta = config.noise_model.beta

    if verbose_3:
        cuda.synchronize()
        t1 = time.perf_counter()

    device = spy.current_device()

    precision_matrices = device.create_texture(
        width=img.width // 2,
        height=img.height // 2,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
    )
    shader = Path(__file__).resolve().parents[1] / "shaders" / "covariance.slang"
    program = device.load_program(str(shader), ["computeCovariance"])
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(
        thread_count=[precision_matrices.width, precision_matrices.height, 1],
        vars={
            "var_noise": 1e-4,  # TODO Placeholder until calibrated noise variance is wired in.
            "covariance_parameters1": spy.float4(
                k_shrink / sigma_detail,  # 1 / sigma_shrink
                1.0 / (k_stretch * sigma_detail),  # 1 / sigma_stretched
                D_th,  # gradient clip
                1.0 / sigma_detail,  # 1/ sigma_base (or sigma_detail)
            ),
            "covariance_parameters2": spy.float4(
                1.0 / (sigma_detail * k_denoise),  # 1 / (sigma_blurry)
                1.0 / D_tr,  # Gradient transition
                0.0,
                0.0,
            ),  # Dummy padding
        },
        sourceRaw=img,
        out=precision_matrices,
    )

    if verbose_3:
        cuda.synchronize()
        t1 = getTime(t1, "- Covariances estimated")

    return precision_matrices
