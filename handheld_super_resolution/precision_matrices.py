"""Estimate the precision matrices used for frame merging."""

from pathlib import Path

import slangpy as spy

from .config import Config, CovarianceType

SEL_HARD_THRESHOLD = 0
SEL_LINEAR = 1


def estimate_precision_matrices(
    img: spy.Texture,
    noise_lut: spy.Texture,
    noise_offset_uv: tuple[float, float],
    noise_scale_uv: tuple[float, float],
    config: Config,
):
    """Return a texture containing one packed precision matrix per Bayer quad."""

    sigma_detail = float(config.merging.covariance.sigma_detail)
    k_denoise = float(config.merging.covariance.k_denoise)
    D_th = float(config.merging.covariance.D_th)
    D_tr = float(config.merging.covariance.D_tr)
    k_stretch = float(config.merging.covariance.k_stretch)
    k_shrink = float(config.merging.covariance.k_shrink)

    device = spy.current_device()

    precision_matrices = device.create_texture(
        width=img.width // 2,
        height=img.height // 2,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
    )
    shader = Path(__file__).resolve().parent / "shaders" / "precision_matrices.slang"
    program = device.load_program(str(shader), ["computePrecisionMatrices"])
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(
        thread_count=[precision_matrices.width, precision_matrices.height, 1],
        vars={
            "is_isotropic": config.merging.covariance_type == CovarianceType.ISO,
            "enable_green_diff": config.merging.covariance.enable_green_diff,
            "noise_correction": config.merging.covariance.noise_correction,
            "noise_offset_uv": spy.float2(*noise_offset_uv),
            "noise_scale_uv": spy.float2(*noise_scale_uv),
            "precision_parameters1": spy.float4(
                k_shrink / sigma_detail,  # 1 / sigma_shrink
                1.0 / (k_stretch * sigma_detail),  # 1 / sigma_stretched
                D_th,  # gradient clip
                1.0 / sigma_detail,  # 1/ sigma_base (or sigma_detail)
            ),
            "precision_parameters2": spy.float4(
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
