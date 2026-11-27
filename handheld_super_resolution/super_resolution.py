import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import slangpy as spy
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter, median_filter, minimum_filter

from . import alignment, raw2rgb
from .config import Config
from .debug_writer import NAN_COLOR, DebugWriter
from .merge import merge
from .noise_lut import create_zero_noise_lut, load_noise_lut
from .noise_profile import expand_noise_profile_to_rgbg
from .precision_matrices import estimate_precision_matrices
from .robustness import compute_robustness, init_robustness
from .unblocker import compute_unblocker_texture
from .utils import (
    DEFAULT_NUMPY_FLOAT_TYPE,
    FINAL_TILE_SIZE,
    FINAL_TILE_STRIDE,
    getTime,
)
from .utils_dng import load_dng_burst
from .utils_image import (
    apply_orientation,
    compute_grey_images,
    estimate_image_snr,
    rggb_to_cfa,
)


def alignment_options(config: Config) -> dict:
    """Translate fine-to-coarse pipeline settings to the alignment API."""
    return {
        "scale_factors": config.alignment.factors,
        "tile_sizes": config.alignment.tile_sizes,
        "search_radii": config.alignment.search_radii,
        "norm": config.alignment.block_matching.metrics,
        "ica_iterations": config.alignment.ica.n_iter,
    }


def main(
    ref_img: NDArray[np.float32],
    comp_imgs: NDArray[np.float32],
    config: Config,
    *,
    source_cfa: NDArray | None = None,
    orientation: int = 1,
) -> tuple[NDArray, NDArray]:
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
    accumulated_rejection : Array[imshape_y//2, imshape_x//2]
        Sum of the comparison-frame rejection masks.

    """

    ### verbose and timing related stuff
    verbose = config.verbose >= 1
    t1 = 0.0
    im_time = 0.0

    debug_writer = (
        DebugWriter(cfa=source_cfa, orientation=orientation) if config.debug else None
    )
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

    if config.requires_noise_lut:
        assert config.noise_model.lut_path is not None
        noise_lut, noise_offset_uv, noise_scale_uv = load_noise_lut(
            device,
            config.noise_model.lut_path,
            config.noise_model.alpha,
            config.noise_model.beta,
        )
    else:
        noise_lut, noise_offset_uv, noise_scale_uv = create_zero_noise_lut(device)

    if verbose:
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()

    #### alignment reference (NumPy gray input)
    ref_grey = compute_grey_images(ref_img, config.alignment.grey_method)
    options = alignment_options(config)
    ref_pyramid = alignment.build_reference_pyramid(ref_grey, **options)
    if debug_writer is not None:
        debug_writer.write_grayscale_pyramid(
            "alignment_reference_grey_pyramid",
            (level.image for level in ref_pyramid.levels),
        )

    #### Local stats estimation
    if config.robustness.enabled:
        ref_local_stats = init_robustness(tx_ref_img, config)
        if debug_writer is not None:
            debug_writer.write_local_stats(ref_local_stats.to_numpy(), reference=True)
    else:
        ref_local_stats = None

    mask_shape = (ref_img.shape[0] // 2, ref_img.shape[1] // 2)
    accumulated_rejection = device.create_texture(
        width=mask_shape[1],
        height=mask_shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
        data=np.zeros(mask_shape, dtype=DEFAULT_NUMPY_FLOAT_TYPE),
        label="Accumulated rejection",
    )

    num, den = init_out_textures(ref_img.shape, config.scale)

    tx_unblocker, unblocker_sampler, unblocker_uv_scale = compute_unblocker_texture(
        ref_img, config, debug_writer
    )

    #### Ref covariance estimation
    ref_precision_matrices = estimate_precision_matrices(
        tx_ref_img, noise_lut, noise_offset_uv, noise_scale_uv, config
    )
    if debug_writer is not None:
        debug_writer.write_precision_matrix(
            "ref_precision", ref_precision_matrices.to_numpy()
        )

    tx_flow, flow_texture_offset_uv, flow_texture_scale_uv = create_flow_texture(
        ref_grey.shape
    )
    unit_rejection_mask = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=ref_grey.shape[1],
        height=ref_grey.shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=(spy.TextureUsage.shader_resource),
        data=np.ones(ref_grey.shape, dtype=DEFAULT_NUMPY_FLOAT_TYPE),
        label="unitRejectionMask",
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
        moving_grey = compute_grey_images(
            comp_imgs[im_id], config.alignment.grey_method
        )
        moving_pyramid = alignment.build_moving_pyramid(moving_grey, **options)
        if debug_writer is not None:
            debug_writer.write_grayscale_pyramid(
                "alignment_moving_grey_pyramid",
                (level.image for level in moving_pyramid.levels),
            )
        flow = alignment.align(ref_pyramid, moving_pyramid)
        if debug_writer is not None:
            debug_writer.write_flow("optical_flow", flow)

        prepare_flow(tx_flow, flow, moving_grey.shape)
        tx_mov_img.copy_from_numpy(comp_imgs[im_id])

        #### Robustness
        if config.robustness.enabled:
            assert ref_local_stats is not None
            rejection_mask = compute_robustness(
                tx_mov_img,
                ref_local_stats,
                tx_flow,
                flow_texture_offset_uv,
                flow_texture_scale_uv,
                noise_lut,
                noise_offset_uv,
                noise_scale_uv,
                tx_unblocker,
                unblocker_sampler,
                unblocker_uv_scale,
                config,
                debug_writer,
                accumulated_rejection=accumulated_rejection,
            )
        else:
            rejection_mask = unit_rejection_mask

        #### Precision matrix estimation
        precision_matrices = estimate_precision_matrices(
            tx_mov_img, noise_lut, noise_offset_uv, noise_scale_uv, config
        )

        if debug_writer is not None:
            debug_writer.write_precision_matrix(
                "precision", precision_matrices.to_numpy()
            )

        #### Merging
        merge(
            tx_mov_img,
            tx_flow,
            flow_texture_offset_uv,
            flow_texture_scale_uv,
            precision_matrices,
            rejection_mask,
            num,
            den,
            accumulated_rejection=accumulated_rejection,
            config=config.merging,
        )

        if verbose:
            getTime(im_time, "\nImage processed (Total)")

    # Comparison alignment overwrites the flow texture. The reference uses zero flow.
    if len(comp_imgs):
        tx_flow.copy_from_numpy(
            np.zeros((tx_flow.height, tx_flow.width, 4), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
        )
    merge(
        tx_ref_img,
        tx_flow,
        flow_texture_offset_uv,
        flow_texture_scale_uv,
        ref_precision_matrices,
        unit_rejection_mask,
        num,
        den,
        accumulated_rejection=accumulated_rejection,
        config=config.merging,
        is_reference=True,
    )

    num = num.to_numpy()[..., :3]
    den = den.to_numpy()[..., :3]
    out = num / den

    accumulated = accumulated_rejection.to_numpy()
    if debug_writer is not None:
        debug_writer.write_scalar_channels("den", den, ("r", "g", "b"))
        debug_writer.write_scalar(
            "accumulated_rejection",
            accumulated,
            value_range=(0.0, float(max(1, len(comp_imgs)))),
        )

    if verbose:
        s = "\nTotal ellapsed time : "
        print(
            s,
            " " * (50 - len(s)),
            ": ",
            round((time.perf_counter() - t1), 2),
            "seconds",
        )

    return out, accumulated


def process(burst_path: Path | str, config: Config) -> tuple[NDArray, NDArray]:
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
    hr_output : Array
        The processed image
    accumulated_rejection : Array
        Sum of the comparison-frame rejection masks in output orientation.

    """
    config.sanitize()
    config = deepcopy(config)

    currentTime, verbose_1, verbose_2 = (
        time.perf_counter(),
        config.verbose >= 1,
        config.verbose >= 2,
    )

    try:
        backend = spy.DeviceType[config.backend]
    except KeyError:
        raise ValueError(
            f"Unknown backend {config.backend!r}. Run --list-devices for choices."
        ) from None

    adapter_luid = None
    if config.device is not None:
        if backend == spy.DeviceType.automatic:
            raise ValueError(
                "Select a backend with --backend when specifying --device."
            )
        adapters = spy.Device.enumerate_adapters(backend)
        if not 0 <= config.device < len(adapters):
            raise ValueError(
                f"No device {config.device} on {backend.name}. Run --list-devices for choices."
            )
        adapter_luid = adapters[config.device].luid

    # reading image stack
    dng_stack = load_dng_burst(burst_path)
    orientation = 1
    if config.postprocessing.orientate_image and "Image Orientation" in dng_stack.tags:
        orientation = dng_stack.tags["Image Orientation"].values[0]
    ref_raw, raw_comp = (
        dng_stack.get_raw_arrays()
    )  # Scale [black level, whiteleve] -> [0, 1] WITHOUT WB, clipping or anything

    if config.noise_model.alpha is not None:
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

    #### Estimating ref image SNR
    if config.force_snr is None:
        snr = estimate_image_snr(
            ref_raw, config.noise_model.alpha, config.noise_model.beta
        )
        if verbose_1:
            snr_db = 20 * np.log10(snr) if snr != 0 else -np.inf
            print(f"Estimated linear SNR: {snr:.2f} ({snr_db:.2f} dB)")
    else:
        snr = config.force_snr

    config.resolve_snr(snr)

    config.validate_alignment_shape(ref_raw.shape)

    if verbose_1:
        print(f"\nRuntime configuration:\n{config.dump()}\n")

    #### Running the handheld pipeline
    HERE = Path(__file__).resolve().parent
    device = spy.Device(
        type=backend,
        adapter_luid=adapter_luid,
        compiler_options={"include_paths": [HERE]},
    )
    with device:
        if verbose_1:
            print(
                f"Backend: {device.info.type.name}, device: {device.info.adapter_name}"
            )
        hr_output, accumulated_rejection = main(
            ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE),
            raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE),
            config,
            source_cfa=dng_stack.cfa,
            orientation=orientation,
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
    accumulated_rejection = rggb_to_cfa(accumulated_rejection, dng_stack.cfa)

    #### post processing
    post_processing_enabled = config.postprocessing.enabled

    if post_processing_enabled:
        if verbose_2:
            print("-- Post processing image")

        hr_output = raw2rgb.postprocess(hr_output, dng_stack, config)

    accumulated_rejection = apply_orientation(accumulated_rejection, orientation)

    return hr_output, accumulated_rejection


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


def init_out_textures(input_imsape: tuple[float, float], scale: float):
    device = spy.current_device()
    input_height, input_width = input_imsape
    output_size = (
        round(scale * input_height),
        round(scale * input_width),
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
    return num, den


def create_flow_texture(imshape: tuple[int, int]):
    height_px, width_px = imshape
    device = spy.current_device()

    flow_height = (height_px - FINAL_TILE_SIZE) // FINAL_TILE_STRIDE + 1
    flow_width = (width_px - FINAL_TILE_SIZE) // FINAL_TILE_STRIDE + 1
    ### The optical_flow is hardcoded on a peculiar layout: FINAL_TILE_SIZE, with a stride of FINAL_TILE_STRIDE.
    # We will ship the flow in a texture, which assumes that d separates 2 vertices, and d/2 separates the first vertex from the edge.
    # We must map the image uv to the flow uv.
    # We crop the extremities of the flow tiles, to make it fit a regular texture, and we use a clamp_to_edge sampler later.
    # The resulting flow texture will cover this many grey pixels
    flow_texture_width_in_gray_pixel = FINAL_TILE_STRIDE * flow_width
    flow_texture_height_in_gray_pixel = FINAL_TILE_STRIDE * flow_height
    flow_texture_scale_u = width_px / flow_texture_width_in_gray_pixel
    flow_texture_scale_v = height_px / flow_texture_height_in_gray_pixel

    # The first tile center is at half the tile size. Point sampling places
    # that center half a stride from the flow texture's left/top edge.
    flow_grid_edge = 0.5 * (FINAL_TILE_SIZE - FINAL_TILE_STRIDE)
    flow_texture_offset_u = flow_grid_edge / width_px
    flow_texture_offset_v = flow_grid_edge / height_px
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
    return tx_flow, flow_texture_offset_uv, flow_texture_scale_uv
