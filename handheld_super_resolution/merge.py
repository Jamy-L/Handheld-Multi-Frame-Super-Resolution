"""
Created on Mon Aug  1 18:38:07 2022

This script contains :
    - The implementation of Alg. 4, the conventionnal accumulation
    - The implementation of Alg. 11, where the reference image is merged


@author: jamyl
"""

from pathlib import Path

import slangpy as spy


def merge(
    comp_img: spy.Texture,
    alignments: spy.Texture,
    flow_offset_uv: tuple[float, float],
    flow_scale_uv: tuple[float, float],
    precision_matrices: spy.Texture,
    r: spy.Texture,
    num: spy.Texture,
    den: spy.Texture,
):
    """
    Implementation of Alg. 4: Accumulation
    Accumulates comp_img (J_n, n>1) into num and den, based on the alignment
    V_n, the covariance matrices Omega_n and the robustness mask estimated before.


    Parameters
    ----------
    comp_imgs : device Array [imsize_y, imsize_x]
        The non-reference image to merge (J_n)
    alignments : device Array[n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment V_n(p)
    covs : device array[imsize_y//2, imsize_x//2, 2, 2]
        covariance matrices Omega_n
    r : Device_Array[imsize_y//2, imsize_x//2]
        Robustness mask r_n
    num : device Array[s*imshape_y, s*imshape_x, c]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x, c]
        Denominator of the accumulator


    Returns
    -------
    None

    """
    device = spy.current_device()
    shader = Path(__file__).resolve().parents[1] / "shaders" / "fusion.slang"
    program = device.load_program(str(shader), ["fusion"])
    kernel = device.create_compute_kernel(program)
    precision_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        address_u=spy.TextureAddressingMode.clamp_to_edge,
        address_v=spy.TextureAddressingMode.clamp_to_edge,
    )
    robustness_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        address_u=spy.TextureAddressingMode.clamp_to_edge,
        address_v=spy.TextureAddressingMode.clamp_to_edge,
    )
    flow_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point,
        mag_filter=spy.TextureFilteringMode.point,
        mip_filter=spy.TextureFilteringMode.point,
        address_u=spy.TextureAddressingMode.clamp_to_edge,
        address_v=spy.TextureAddressingMode.clamp_to_edge,
    )
    kernel.dispatch(
        thread_count=[num.width, num.height, 1],
        sourceRaw=comp_img,
        alignments=alignments,
        flowSampler=flow_sampler,
        precision=precision_matrices,
        precisionSampler=precision_sampler,
        robustness=r,
        robustnessSampler=robustness_sampler,
        numerator=num,
        denominator=den,
        flowOffsetUV=spy.float2(*flow_offset_uv),
        flowScaleUV=spy.float2(*flow_scale_uv),
    )
