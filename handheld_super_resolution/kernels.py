"""
Created on Sat Aug  6 17:26:48 2022

This script contains the implementation of Alg. 5: ComputeKernelCovariance,
where the merge kernels are estimated


@author: jamyl
"""

from pathlib import Path

import slangpy as spy

from .config import Config, KernelType

SEL_HARD_THRESHOLD = 0
SEL_LINEAR = 1


def estimate_precision_matrices(
    img: spy.Texture,
    noise_lut: spy.Texture,
    noise_offset_uv: tuple[float, float],
    noise_scale_uv: tuple[float, float],
    config: Config,
):
    """
    Implementation of Alg. 5: ComputeKernelCovariance
    Returns the kernels covariance matrices for the frame J_n, sampled at the
    center of every Bayer quad.

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

    sigma_detail = float(config.merging.kernel.sigma_detail)
    k_denoise = float(config.merging.kernel.k_denoise)
    D_th = float(config.merging.kernel.D_th)
    D_tr = float(config.merging.kernel.D_tr)
    k_stretch = float(config.merging.kernel.k_stretch)
    k_shrink = float(config.merging.kernel.k_shrink)

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
            "is_isotropic": config.merging.kernel_type == KernelType.ISO,
            "enable_green_diff": config.merging.kernel.enable_green_diff,
            "noise_correction": config.merging.kernel.noise_correction,
            "noise_offset_uv": spy.float2(*noise_offset_uv),
            "noise_scale_uv": spy.float2(*noise_scale_uv),
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
        noise_lut=noise_lut,
        noiseSampler=device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        ),
        out=precision_matrices,
    )

    return precision_matrices
