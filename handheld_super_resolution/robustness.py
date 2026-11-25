"""
Created on Fri Sep  9 09:00:17 2022

This script contains :
    - The implementation of Algorithm 6: ComputeRobustness
    - The implementation of Algorithm 7: ComputeGuideImage
    - The implementation of Algorithm 8: ComputeLocalStatistics
    - The implementation of Algorithm 9: ComputeLocalMin


@author: jamyl
"""

import time
from pathlib import Path

from cv2 import line
import slangpy as spy

from .config import Config, LocalStatsWarpFilter
from .debug_writer import DebugWriter
from .utils import (
    timer,
)


# Matches LocalStatsWarpFilter in local_stats_warp_filter.slang.
_LOCAL_STATS_WARP_FILTER_IDS = {
    LocalStatsWarpFilter.BILINEAR: 0,
    LocalStatsWarpFilter.BEZIER: 1,
    LocalStatsWarpFilter.BIQUADRATIC: 2,
}


def init_robustness(ref_img: spy.Texture, config: Config) -> spy.Texture:
    """
    Initialiazes the robustness etimation procedure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Raw reference image J_1
    config : Config
        parameters.

    Returns
    -------
    local_means : device Array[imshape_y, imshape_x, channels]
        local means of the reference image.

    local_stds : device Array[imshape_y, imshape_x, channels]
        local standard deviations of the reference image.

    """
    verbose_3 = config.verbose >= 3

    compute_local_stats_ = timer(
        compute_local_stats, verbose_3, end_s=" - Local stats estimated"
    )

    local_stats = compute_local_stats_(ref_img, config.robustness.k_green_diff_impact)
    local_stats_warp_filter = config.robustness.local_stats_warp_filter
    if local_stats_warp_filter == LocalStatsWarpFilter.BILINEAR:
        return local_stats  # Warping with 0 flow yields identity with bilinear

    return blur_local_stats(local_stats, local_stats_warp_filter)


def blur_local_stats(
    local_stats: spy.Texture, local_stats_warp_filter: LocalStatsWarpFilter
) -> spy.Texture:
    device = spy.current_device()
    shader = Path(__file__).resolve().parents[1] / "shaders" / "guide_image.slang"
    filtered = device.create_texture(
        width=local_stats.width,
        height=local_stats.height,
        format=local_stats.format,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
    )
    clamp_to_edge = spy.TextureAddressingMode.clamp_to_edge
    linear_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        address_u=clamp_to_edge,
        address_v=clamp_to_edge,
    )
    program = device.load_program(str(shader), ["blurLocalStats"])
    device.create_compute_kernel(program).dispatch(
        thread_count=[local_stats.width, local_stats.height, 1],
        local_stats_warp_filter=_LOCAL_STATS_WARP_FILTER_IDS[local_stats_warp_filter],
        linearSampler=linear_sampler,
        stats=local_stats,
        out=filtered,
    )
    return filtered


def compute_robustness(
    comp_img: spy.Texture,
    ref_local_stats: spy.Texture,
    flow: spy.Texture,
    flow_offset_uv: tuple[float, float],
    flow_scale_uv: tuple[float, float],
    noise_texture: spy.Texture,
    noise_offset_uv: tuple[float, float],
    noise_scale_uv: tuple[float, float],
    config: Config,
    debug_writer: DebugWriter | None = None,
) -> spy.Texture:
    """
    this is the implementation of Algorithm 6: ComputeRobustness
    Returns the robustnesses of the compared image J_n (n>1), based on the
    provided flow V_n(p) and the local statistics of the reference frame.

    Parameters
    ----------
    comp_img : device Array[imsize_y, imsize_x]
        Compared raw image J_n (n>1).
    ref_local_means : device Array[imsize_y, imsize_x, c]
        Local means of the reference image
    ref_local_stds : device Array[imsize_y, imsize_x, c]
        Local standard deviations of the reference image
    flows : device Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image V_n(p)
    config : Config
        parameters.
    debug_writer : DebugWriter, optional
        Streams intermediate guide images to disk when provided.

    Returns
    -------
    r : device Array[imsize_y, imsize_x]
        Locally minimized Robustness map, sampled at the center of
        every bayer quad
    """
    current_time, verbose_3 = time.perf_counter(), config.verbose >= 3

    compute_local_stats_ = timer(
        compute_local_stats, verbose_3, end_s=" - Local stats estimated"
    )
    compute_rejection_coeff_ = timer(
        compute_rejection_coeff, verbose_3, end_s=" - Rejection coeffs evaluated"
    )

    # Computing guide image
    local_stats = compute_local_stats_(comp_img, config.robustness.k_green_diff_impact)

    if debug_writer is not None:
        debug_writer.write_local_stats(local_stats.to_numpy())

    rejection_mask = compute_rejection_coeff_(
        ref_local_stats,
        local_stats,
        flow,
        flow_offset_uv,
        flow_scale_uv,
        noise_texture,
        noise_offset_uv,
        noise_scale_uv,
        config,
    )

    if debug_writer is not None:
        r = rejection_mask.to_numpy()
        debug_writer.write_scalar("robustness", r, value_range=(0.0, 1.0))

    rejection_mask = mask_dilation(rejection_mask, config)

    if debug_writer is not None:
        r = rejection_mask.to_numpy()
        debug_writer.write_scalar("robustness_dilated", r, value_range=(0.0, 1.0))

    return rejection_mask


def mask_dilation(rejection_mask: spy.Texture, config: Config) -> spy.Texture:
    device = spy.current_device()
    output = device.create_texture(
        width=rejection_mask.width,
        height=rejection_mask.height,
        format=rejection_mask.format,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    shader = Path(__file__).resolve().parents[1] / "shaders" / "5x5min.slang"
    program = device.load_program(str(shader), ["min5x5"])
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(
        thread_count=[output.width, output.height, 1],
        source=rejection_mask,
        out=output,
    )

    return output


def compute_local_stats(
    raw_img: spy.Texture,
    k_green_diff_impact: float,
) -> spy.Texture:
    device = spy.current_device()

    output = device.create_texture(
        width=raw_img.width // 2,
        height=raw_img.height // 2,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
    )
    shader = Path(__file__).resolve().parents[1] / "shaders" / "guide_image.slang"
    program = device.load_program(str(shader), ["computeGuide"])
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(
        thread_count=[output.width, output.height, 1],
        vars={"k_green_diff_impact": k_green_diff_impact},
        sourceRaw=raw_img,
        out=output,
    )
    return output


def compute_rejection_coeff(
    ref_local_stats: spy.Texture,
    local_stats: spy.Texture,
    flow: spy.Texture,
    flow_offset_uv: tuple[float, float],
    flow_scale_uv: tuple[float, float],
    noise_texture: spy.Texture,
    noise_offset_uv: tuple[float, float],
    noise_scale_uv: tuple[float, float],
    config: Config,
) -> spy.Texture:

    device = spy.current_device()
    output = device.create_texture(
        width=ref_local_stats.width,
        height=ref_local_stats.height,
        format=spy.Format.r32_float,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
    )
    clamp_to_edge = spy.TextureAddressingMode.clamp_to_edge
    flow_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point,
        mag_filter=spy.TextureFilteringMode.point,
        address_u=clamp_to_edge,
        address_v=clamp_to_edge,
    )
    linear_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        address_u=clamp_to_edge,
        address_v=clamp_to_edge,
    )
    shader = Path(__file__).resolve().parents[1] / "shaders" / "rejection_mask.slang"
    program = device.load_program(str(shader), ["computeRejectionCoeff"])
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(
        thread_count=[output.width, output.height, 1],
        vars={
            "noise_correction": config.robustness.noise_correction,
            "noise_offset_uv": spy.float2(*noise_offset_uv),
            "noise_scale_uv": spy.float2(*noise_scale_uv),
            "rejection_params": spy.float4(
                config.robustness.R_mult,
                config.robustness.R_threshold,
                config.robustness.M_th,
                config.robustness.k_std,
            ),
            "R_boost": config.robustness.R_boost,
        },
        local_stats_warp_filter=_LOCAL_STATS_WARP_FILTER_IDS[
            config.robustness.local_stats_warp_filter
        ],
        sourceLocalStats=ref_local_stats,
        localStats=local_stats,
        linearSampler=linear_sampler,
        noise_lut=noise_texture,
        noiseSampler=linear_sampler,
        flow=flow,
        flowSampler=flow_sampler,
        flow_scale_uv=spy.float2(*flow_scale_uv),
        flow_offset_uv=spy.float2(*flow_offset_uv),
        out=output,
    )
    return output
