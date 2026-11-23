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

import numpy as np
import slangpy as spy

from .config import Config
from .debug_writer import DebugWriter
from .utils import (
    timer,
)


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

    local_stats = compute_local_stats_(ref_img)

    return local_stats


def compute_robustness(
    comp_img: spy.Texture,
    ref_local_stats: spy.Texture,
    flow: spy.Texture,
    flow_offset_uv: tuple[float, float],
    flow_scale_uv: tuple[float, float],
    noise_estimatss: spy.Texture,
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
    local_stats = compute_local_stats_(comp_img)

    if debug_writer is not None:
        guide = local_stats.to_numpy()
        # The shader stores (Y, Cg, Co, std) and sums the two green samples.
        y, cg, co = guide[..., 0], guide[..., 1], guide[..., 2]
        frame = np.stack((y - cg + co, y + cg, y - cg - co), axis=-1)
        debug_writer.write_rgb("rgb_guides", frame)
        std = guide[..., 3]
        upper = std[np.isfinite(std)].max(initial=0.0)
        debug_writer.write_scalar("guide_std", std, value_range=(0.0, upper or 1.0))

    rejection_mask = compute_rejection_coeff_(
        ref_local_stats, local_stats, flow, flow_offset_uv, flow_scale_uv, config
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


def compute_local_stats(raw_img: spy.Texture) -> spy.Texture:
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
    stats_sampler = device.create_sampler(
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
            # Placeholder values for the noise model and rejection tuning.
            "var_noise": 0.0,
            "rejection_params": spy.float4(1.0, 0.0, 0.0, 0.0),
        },
        sourceLocalStats=ref_local_stats,
        localStats=local_stats,
        statsSampler=stats_sampler,
        flow=flow,
        flowSampler=flow_sampler,
        flow_scale_uv=spy.float2(*flow_scale_uv),
        flow_offset_uv=spy.float2(*flow_offset_uv),
        out=output,
    )
    return output
