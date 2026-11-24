import time
import warnings
from pathlib import Path

import numpy as np
import slangpy as spy
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter, median_filter, minimum_filter

from . import alignment, raw2rgb
from .config import Config
from .debug_writer import NAN_COLOR, DebugWriter
from .kernels import estimate_precision_matrices
from .merge import merge
from .noise_lut import create_zero_noise_lut, load_noise_lut
from .params import runtime_config, sanitize_config, update_snr_config
from .robustness import compute_robustness, init_robustness
from .utils import (
    DEFAULT_NUMPY_FLOAT_TYPE,
    FINAL_TILE_SIZE,
    FINAL_TILE_STRIDE,
    getTime,
    timer,
)
from .utils_dng import expand_noise_profile_to_rgbg, load_dng_burst
from .utils_image import (
    apply_orientation,
    compute_grey_images,
    estimate_image_snr,
    rggb_to_cfa,
)


def alignment_options(config: Config) -> dict:
    """Translate fine-to-coarse pipeline settings to the native API."""
    return {
        "scale_factors": config.alignment.factors,
        "tile_sizes": config.alignment.tile_sizes,
        "search_radii": config.alignment.search_radii,
        "norm": config.alignment.block_matching.metrics,
        "ica_iterations": config.alignment.ica.n_iter,
    }


def main(
    ref_img: NDArray[np.float32], comp_imgs: NDArray[np.float32], config: Config
) -> tuple[NDArray, dict[str, NDArray]]:
    """
    This is the implementation of Alg. 1: HandheldBurstSuperResolution.
    Some part of Alg. 2: Registration are also integrated for optimisation.

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Reference frame J_1
    comp_imgs : Array[N-1, imshape_y, imshape_x]
        Remaining frames of the burst J_2, ..., J_N

    config : Config
        parameters.

    Returns
    -------
    num : device Array[imshape_y*s, imshape_y*s, 3]
        generated RGB image WITHOUT any post-processing.
    debug_dict : dict
        Contains the accumulated robustness map when requested. Per-frame
        diagnostics are streamed to disk when debugging is enabled.

    """

    grey_method = config.alignment.grey_method

    ### verbose and timing related stuff
    verbose = config.verbose >= 1
    verbose_2 = config.verbose >= 2
    verbose_3 = config.verbose >= 3
    t1 = 0.0
    im_time = 0.0

    init_robustness_ = timer(
        init_robustness,
        verbose_2,
        "\nEstimating ref image local stats",
        "Local stats estimated (Total)",
    )
    compute_grey_images_ = timer(
        compute_grey_images,
        verbose_3,
        end_s=f"- Grey images estimated by {grey_method.value}",
    )
    compute_robustness_ = timer(
        compute_robustness,
        verbose_2,
        "\nEstimating robustness",
        "Robustness estimated (Total)",
    )
    estimate_precision_matrices_ = timer(
        estimate_precision_matrices,
        verbose_2,
        "\nEstimating kernels",
        "Kernels estimated (Total)",
    )
    merge_ = timer(
        merge, verbose_2, "\nAccumulating Image", "Image accumulated (Total)"
    )
    init_alignment_ = timer(
        alignment.build_reference_pyramid,
        verbose_2,
        "\nInitializing alignment",
        "Alignment initialized (Total)",
    )
    align_ = timer(
        alignment.align, verbose_2, "\nBeginning alignment", "Image aligned (Total)"
    )

    bayer_mode = config.mode == "bayer"
    debug_dict = {}
    debug_writer = DebugWriter() if config.debug else None
    if debug_writer is not None:
        # The fixed reference pyramid shares frame 0 with the first comparison.
        debug_writer.next_frame()

    #### Moving to GPU
    device = spy.current_device()
    tx_ref_img = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=ref_img.shape[1],
        height=ref_img.shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=(spy.TextureUsage.shader_resource),
        sampler=device.create_sampler(
            min_filter=spy.TextureFilteringMode.point,
            mag_filter=spy.TextureFilteringMode.point,
            mip_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        ),
        data=ref_img,
        label="ref_image",
    )
    tx_mov_img = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=ref_img.shape[1],
        height=ref_img.shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=(
            spy.TextureUsage.shader_resource
            | spy.TextureUsage.copy_destination  # copy_dest to overrid from_numpy later
        ),
        sampler=device.create_sampler(
            min_filter=spy.TextureFilteringMode.point,
            mag_filter=spy.TextureFilteringMode.point,
            mip_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        ),
        data=None,
        label="mov_image",
    )

    if config.robustness.enabled and config.robustness.noise_correction:
        lut_path = config.noise_model.lut_path
        assert lut_path is not None, (
            "robustness noise correction requires --noise-model.lut-path PATH"
        )
        noise_lut, noise_offset_uv, noise_scale_uv = load_noise_lut(
            device, lut_path, config.noise_model.alpha, config.noise_model.beta
        )
    else:
        noise_lut, noise_offset_uv, noise_scale_uv = create_zero_noise_lut(device)

    if verbose:
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()

    #### alignment reference (NumPy gray input)
    ref_grey = compute_grey_images_(ref_img, grey_method) if bayer_mode else ref_img
    options = alignment_options(config)
    ref_pyramid = init_alignment_(ref_grey, **options)
    if debug_writer is not None:
        debug_writer.write_grayscale_pyramid(
            "alignment_reference_grey_pyramid",
            (level.image for level in ref_pyramid.levels),
        )

    #### Local stats estimation
    if config.robustness.enabled:
        ref_local_stats = init_robustness_(tx_ref_img, config)
    else:
        ref_local_stats = None

    accumulated_r = None
    if config.robustness.save_mask and config.robustness.enabled:
        assert ref_local_stats
        # accumulated_r = cuda.to_device(
        #     np.zeros((ref_local_stats.height, ref_local_stats.width))
        # )

    native_imshape_y, native_imshape_x = ref_img.shape
    output_size = (
        round(config.scale * native_imshape_y),
        round(config.scale * native_imshape_x),
    )

    num = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=output_size[1],
        height=output_size[0],
        format=spy.Format.rgba32_float,
        mip_count=1,
        usage=(spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
        data=np.zeros((*output_size, 4), dtype=DEFAULT_NUMPY_FLOAT_TYPE),
        label="num",
    )
    den = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=output_size[1],
        height=output_size[0],
        format=spy.Format.rgba32_float,
        mip_count=1,
        usage=(spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
        data=np.zeros((*output_size, 4), dtype=DEFAULT_NUMPY_FLOAT_TYPE),
        label="den",
    )

    #### Ref covariance estimation
    precision_matrices = estimate_precision_matrices_(
        tx_ref_img, noise_lut, noise_offset_uv, noise_scale_uv, config
    )
    if debug_writer is not None:
        debug_writer.write_precision_matrix(
            "ref_precision", precision_matrices.to_numpy()
        )

    ##### Merge ref
    flow_height = (ref_grey.shape[0] - FINAL_TILE_SIZE) // FINAL_TILE_STRIDE + 1
    flow_width = (ref_grey.shape[1] - FINAL_TILE_SIZE) // FINAL_TILE_STRIDE + 1
    ### The optical_flow is hardcoded on a peculiar layout: FINAL_TILE_SIZE, with a stride of FINAL_TILE_STRIDE.
    # We will ship the flow in a texture, which assumes that d separates 2 vertices, and d/2 separates the first vertex from the edge.
    # We must map the image uv to the flow uv.
    # We crop the extremities of the flow tiles, to make it fit a regular texture, and we use a clamp_to_edge sampler later.
    # The resulting flow texture will cover this many grey pixels
    flow_texture_width_in_gray_pixel = FINAL_TILE_STRIDE * flow_width
    flow_texture_height_in_gray_pixel = FINAL_TILE_STRIDE * flow_height
    flow_texture_scale_u = ref_grey.shape[1] / flow_texture_width_in_gray_pixel
    flow_texture_scale_v = ref_grey.shape[0] / flow_texture_height_in_gray_pixel

    # The first tile center is at half the tile size. Point sampling places
    # that center half a stride from the flow texture's left/top edge.
    flow_grid_edge = 0.5 * (FINAL_TILE_SIZE - FINAL_TILE_STRIDE)
    flow_texture_offset_u = flow_grid_edge / ref_grey.shape[1]
    flow_texture_offset_v = flow_grid_edge / ref_grey.shape[0]
    # a location uv on an image is (uv - offset) * scale on the flow's uv
    flow_texture_scale_uv = (flow_texture_scale_u, flow_texture_scale_v)
    flow_texture_offset_uv = (flow_texture_offset_u, flow_texture_offset_v)

    tx_flow = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=flow_width,
        height=flow_height,
        format=spy.Format.rgba32_float,
        mip_count=1,
        usage=(spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination),
        data=np.zeros(
            (flow_height, flow_width, 4), dtype=DEFAULT_NUMPY_FLOAT_TYPE
        ),  # Init with 0 for ref
        label="flow",
    )
    rejection_mask = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=ref_grey.shape[1],
        height=ref_grey.shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=(spy.TextureUsage.shader_resource),
        data=np.ones(
            ref_grey.shape, dtype=DEFAULT_NUMPY_FLOAT_TYPE
        ),  # Init with 1 for ref
        label="rejectionMask",
    )
    merge_(
        tx_ref_img,
        tx_flow,
        flow_texture_offset_uv,
        flow_texture_scale_uv,
        precision_matrices,
        rejection_mask,
        num,
        den,
        config,
    )

    if verbose:
        getTime(t1, "\nRef Img processed (Total)")

    for im_id in range(comp_imgs.shape[0]):
        if debug_writer is not None and im_id > 0:
            debug_writer.next_frame()

        if verbose:
            print(f"\nProcessing image {im_id + 1} ---------\n")
            im_time = time.perf_counter()

        #### CPU alignment (NumPy in, NumPy out)
        moving_grey = (
            compute_grey_images_(comp_imgs[im_id], grey_method)
            if bayer_mode
            else comp_imgs[im_id]
        )
        moving_pyramid = alignment.build_moving_pyramid(moving_grey, **options)
        if debug_writer is not None:
            debug_writer.write_grayscale_pyramid(
                "alignment_moving_grey_pyramid",
                (level.image for level in moving_pyramid.levels),
            )
        flow = align_(ref_pyramid, moving_pyramid)
        if debug_writer is not None:
            debug_writer.write_flow("optical_flow", flow)

        prepare_flow(tx_flow, flow, moving_grey.shape)
        tx_mov_img.copy_from_numpy(comp_imgs[im_id])

        #### Robustness
        if config.robustness.enabled:
            assert ref_local_stats is not None
            rejection_mask = compute_robustness_(
                tx_mov_img,
                ref_local_stats,
                tx_flow,
                flow_texture_offset_uv,
                flow_texture_scale_uv,
                noise_lut,
                noise_offset_uv,
                noise_scale_uv,
                config,
                debug_writer,
            )
        else:
            # robustness is inited with ones for ref.
            # Keep it
            pass

        if accumulated_r is not None:
            # TODO port this
            # add(accumulated_r, robustness)
            pass

        #### Kernel estimation
        precision_matrices = estimate_precision_matrices_(
            tx_mov_img, noise_lut, noise_offset_uv, noise_scale_uv, config
        )

        if debug_writer is not None:
            debug_writer.write_precision_matrix(
                "precision", precision_matrices.to_numpy()
            )

        #### Merging
        merge_(
            tx_mov_img,
            tx_flow,
            flow_texture_offset_uv,
            flow_texture_scale_uv,
            precision_matrices,
            rejection_mask,
            num,
            den,
            config,
        )

        if verbose:
            getTime(im_time, "\nImage processed (Total)")

    num = num.to_numpy()[..., :3]
    den = den.to_numpy()[..., :3]
    out = num / den

    if debug_writer is not None:
        debug_writer.write_scalar_channels("den", den, ("r", "g", "b"))

    if verbose:
        s = "\nTotal ellapsed time : "
        print(
            s,
            " " * (50 - len(s)),
            ": ",
            round((time.perf_counter() - t1), 2),
            "seconds",
        )

    if config.robustness.save_mask and config.robustness.enabled and accumulated_r:
        debug_dict["accumulated robustness"] = accumulated_r.copy_to_host()

    return out, debug_dict


def process(burst_path: Path | str, config: Config):
    """
    Processes the burst

    Parameters
    ----------
    burst_path : str or Path
        Path of the folder where the .dng burst is located
    config : Config
        parameters.

    Returns
    -------
    Array
        The processed image

    """
    config = runtime_config(config)

    currentTime, verbose_1, verbose_2 = (
        time.perf_counter(),
        config.verbose >= 1,
        config.verbose >= 2,
    )

    # reading image stack
    dng_stack = load_dng_burst(burst_path)
    ref_raw, raw_comp = (
        dng_stack.get_raw_arrays()
    )  # Scale [black level, whiteleve] -> [0, 1] WITHOUT WB, clipping or anything

    if config.noise_model.alpha is not None:
        if config.noise_model.beta is None:
            raise ValueError("If alpha is provided, beta must also be provided.")
        # User provided custom values.
        print("Using user-provided alpha and beta values")
        config.noise_model.alpha, config.noise_model.beta = (
            expand_noise_profile_to_rgbg(
                config.noise_model.alpha, config.noise_model.beta
            )
        )
    else:
        # Extract alpha and beta from the DNG tags.
        config.noise_model.alpha = dng_stack.alpha
        config.noise_model.beta = dng_stack.beta

    if verbose_2:
        currentTime = getTime(currentTime, " -- Read raw files")

    #### Estimating ref image SNR
    if config.force_snr is None:
        snr = estimate_image_snr(
            ref_raw, config.noise_model.alpha, config.noise_model.beta
        )
        if verbose_1:
            print(f"Estimated snr: {snr:.2f} dB")
    else:
        snr = config.force_snr

    update_snr_config(config, snr)

    sanitize_config(config, ref_raw.shape)

    if verbose_1:
        print(f"\nRuntime configuration:\n{config.dump()}\n")

    #### Running the handheld pipeline
    import slangpy as spy

    HERE = Path(__file__).resolve().parent
    backend = spy.DeviceType.vulkan
    device = spy.Device(
        type=backend,
        adapter_luid=None,
        compiler_options={"include_paths": [HERE]},
    )
    with device:
        hr_output, debug_dict = main(
            ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE),
            raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE),
            config,
        )

    if np.any(np.isnan(hr_output)):
        mask = np.any(np.isnan(hr_output), axis=-1)
        warnings.warn(f"Found {mask.sum()} pixels with NaNs in output")
        hr_output[mask] = (
            np.array(NAN_COLOR, DEFAULT_NUMPY_FLOAT_TYPE) / 255.0
            if config.debug
            else [0.0, 0.0, 0.0]
        )

    if np.any(np.isinf(hr_output)):
        warnings.warn(f"Found {(np.isinf(hr_output)).sum()} Inf in output")

    #### Deflip the image
    hr_output = np.moveaxis(hr_output, -1, 0)
    hr_output = rggb_to_cfa(hr_output, dng_stack.cfa)
    hr_output = np.moveaxis(hr_output, 0, -1)
    if "accumulated robustness" in debug_dict:
        debug_dict["accumulated robustness"] = rggb_to_cfa(
            debug_dict["accumulated robustness"], dng_stack.cfa
        )

    #### post processing
    post_processing_enabled = config.postprocessing.enabled

    if post_processing_enabled:
        if verbose_2:
            print("-- Post processing image")

        hr_output = raw2rgb.postprocess(hr_output, dng_stack, config)

    if (
        "accumulated robustness" in debug_dict
        and config.postprocessing.orientate_image
        and "Image Orientation" in dng_stack.tags
    ):
        ori = dng_stack.tags["Image Orientation"].values[0]
        debug_dict["accumulated robustness"] = apply_orientation(
            debug_dict["accumulated robustness"], ori
        )

    return hr_output, debug_dict


def prepare_flow(
    texture: spy.Texture, flow: NDArray[np.float32], guide_imshape: tuple[int, int]
) -> None:
    if flow.shape != (texture.height, texture.width, 2):
        raise ValueError("Flow must match the texture dimensions and have two channels")

    # Normalise flow in uv units
    height, width = guide_imshape
    flow_uv = flow / np.array([width, height], dtype=np.float32)

    # median_filter
    filtered_flow = median_filter(flow_uv, size=(3, 3, 1), mode="nearest")

    # Filter only the spatial axes, with the outermost tiles repeated at the edges.
    spread = maximum_filter(
        filtered_flow, size=(3, 3, 1), mode="nearest"
    ) - minimum_filter(filtered_flow, size=(3, 3, 1), mode="nearest")
    data = np.zeros((texture.height, texture.width, 4), dtype=np.float32)
    data[..., :2] = filtered_flow
    data[..., 2] = np.hypot(spread[..., 0], spread[..., 1])
    texture.copy_from_numpy(data)
