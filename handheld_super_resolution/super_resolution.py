import time
import warnings
from pathlib import Path

import numpy as np
import slangpy as spy
from numba import cuda
from numpy.typing import NDArray

from . import alignment, raw2rgb
from .config import Config
from .debug_writer import NAN_COLOR, DebugWriter
from .kernels import estimate_kernels
from .merge import merge
from .noise_lut import load_noise_lut
from .params import runtime_config, sanitize_config, update_snr_config
from .robustness import compute_robustness, init_robustness
from .utils import (
    DEFAULT_NUMPY_FLOAT_TYPE,
    FINAL_TILE_SIZE,
    FINAL_TILE_STRIDE,
    add,
    divide,
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


def main(ref_img: NDArray[np.float32], comp_imgs: NDArray[np.float32], config: Config) -> tuple[NDArray, dict[str, NDArray]]:
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
    t1 = 0.
    im_time = 0.

    init_robustness_ = timer(init_robustness, verbose_2, "\nEstimating ref image local stats", 'Local stats estimated (Total)')
    compute_grey_images_ = timer(compute_grey_images, verbose_3, end_s=f"- Grey images estimated by {grey_method.value}")
    compute_robustness_ = timer(compute_robustness, verbose_2, '\nEstimating robustness', 'Robustness estimated (Total)')
    estimate_kernels_ = timer(estimate_kernels, verbose_2, '\nEstimating kernels', 'Kernels estimated (Total)')
    merge_ = timer(merge, verbose_2, '\nAccumulating Image', 'Image accumulated (Total)')
    divide_ = timer(divide, verbose_2, end_s='\n------------------------\nImage normalized (Total)')
    init_alignment_ = timer(alignment.build_reference_pyramid, verbose_2, '\nInitializing alignment', 'Alignment initialized (Total)')
    align_ = timer(alignment.align, verbose_2, '\nBeginning alignment', 'Image aligned (Total)')

    bayer_mode = config.mode=='bayer'
    debug_dict = {}
    debug_writer = DebugWriter() if config.debug else None
    if debug_writer is not None:
        # The fixed reference pyramid shares frame 0 with the first comparison.
        debug_writer.next_frame()

    if config.robustness.enabled and config.robustness.noise_correction:
        curve_size = len(config.noise_model.sigma_sq_curve)
        if curve_size < 2 or len(config.noise_model.d_sq_curve) != curve_size:
            raise ValueError(
                "noise-correction curves were not initialized; call process() with a "
                "validated noise_model.lut_path"
            )

    #### Moving to GPU
    cuda_ref_img = cuda.to_device(ref_img)
    device = spy.current_device()
    tx_ref_img = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=ref_img.shape[1],
        height=ref_img.shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=(
            spy.TextureUsage.shader_resource
        ),
        sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.point,
            mag_filter=spy.TextureFilteringMode.point,
            mip_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        ),
        data=ref_img,
        label="ref_image",
    )

    # This running buffer is for the image being processed
    stream = cuda.stream()
    cuda_img = cuda.device_array_like(comp_imgs[0], stream=stream)
    cuda.synchronize()
    cuda_sigma_sq_curve = cuda.to_device(np.asarray(config.noise_model.sigma_sq_curve, dtype=np.float32))
    cuda_d_sq_curve = cuda.to_device(np.asarray(config.noise_model.d_sq_curve, dtype=np.float32))
    
    if verbose :
        print("\nProcessing reference image ---------\n")
        t1 = time.perf_counter()

    #### alignment reference (NumPy gray input)
    ref_grey = compute_grey_images_(ref_img, grey_method) if bayer_mode else ref_img
    options = alignment_options(config)
    ref_pyramid = init_alignment_(ref_grey, **options)
    if debug_writer is not None:
        debug_writer.write_grayscale_pyramid(
            "alignment_reference_grey_pyramid", (level.image for level in ref_pyramid.levels)
        )

    #### Local stats estimation
    if config.robustness.enabled:
        tx_local_stats = init_robustness_(tx_ref_img, config)
    else:
        tx_local_stats = None

    
    accumulated_r = None
    if config.robustness.save_mask and config.robustness.enabled:
        assert tx_local_stats
        accumulated_r = cuda.to_device(np.zeros((tx_local_stats.height, tx_local_stats.width)))

    native_imshape_y, native_imshape_x = ref_img.shape
    output_size = (
        round(config.scale*native_imshape_y),
        round(config.scale*native_imshape_x))
    
    num = cuda.to_device(np.zeros((*output_size, 3), dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    den = cuda.to_device(np.zeros((*output_size, 3), dtype = DEFAULT_NUMPY_FLOAT_TYPE))

    #### Ref kernel estimation
    cuda_kernels = estimate_kernels_(tx_ref_img, config)
    
    ##### Merge ref
    dummy_alignment = cuda.to_device(np.zeros(((ref_grey.shape[0] - FINAL_TILE_SIZE) // FINAL_TILE_STRIDE + 1,
                                              (ref_grey.shape[1] - FINAL_TILE_SIZE) // FINAL_TILE_STRIDE + 1, 2),
                                             dtype=DEFAULT_NUMPY_FLOAT_TYPE))
    dummy_r = cuda.to_device(np.ones(cuda_ref_img.shape, dtype = DEFAULT_NUMPY_FLOAT_TYPE))
    merge_(cuda_ref_img, dummy_alignment, cuda_kernels, dummy_r, num, den, config)
    
    if verbose :
        cuda.synchronize()
        getTime(t1, '\nRef Img processed (Total)')


    for im_id in range(comp_imgs.shape[0]):
        if debug_writer is not None and im_id > 0:
            debug_writer.next_frame()

        if verbose :
            cuda.synchronize()
            print(f"\nProcessing image {im_id+1} ---------\n")
            im_time = time.perf_counter()
        
        #### Moving to GPU
        cuda.to_device(comp_imgs[im_id], to=cuda_img, stream=stream)
        
        #### CPU alignment (NumPy in, NumPy out)
        moving_grey = (compute_grey_images_(comp_imgs[im_id], grey_method)
                       if bayer_mode else comp_imgs[im_id])
        moving_pyramid = alignment.build_moving_pyramid(moving_grey, **options)
        if debug_writer is not None:
            debug_writer.write_grayscale_pyramid(
                "alignment_moving_grey_pyramid", (level.image for level in moving_pyramid.levels)
            )
        flow = align_(ref_pyramid, moving_pyramid)
        device_flow = cuda.to_device(flow)

        if debug_writer is not None:
            debug_writer.write_flow("optical_flow", flow)

        #### Robustness
        if config.robustness.enabled:
            assert ref_local_means is not None
            assert ref_local_stds is not None
            robustness = compute_robustness_(
                cuda_img, ref_local_means, ref_local_stds, device_flow,
                (cuda_sigma_sq_curve, cuda_d_sq_curve), config, debug_writer,
            )
        else:
            temp = np.ones_like(cuda_img, DEFAULT_NUMPY_FLOAT_TYPE)
            robustness = cuda.to_device(temp)

        if accumulated_r is not None:
            add(accumulated_r, robustness)
        
        #### Kernel estimation
        cuda_kernels = estimate_kernels_(cuda_img, config)
        
        #### Merging
        merge_(cuda_img, device_flow, cuda_kernels, robustness, num, den, config)
        
        if verbose :
            cuda.synchronize()
            getTime(im_time, '\nImage processed (Total)')
            
        stream.synchronize()


        
    # num is outwritten into num/den
    divide_(num, den)

    if debug_writer is not None:
        debug_writer.write_scalar_channels(
            "den", den.copy_to_host(), ("r", "g", "b")
        )
    
    if verbose :
        s = '\nTotal ellapsed time : '
        print(s, ' ' * (50 - len(s)), ': ', round((time.perf_counter() - t1), 2), 'seconds')
    
    if config.robustness.save_mask and config.robustness.enabled and accumulated_r:
        debug_dict['accumulated robustness'] = accumulated_r.copy_to_host()
        
    return num.copy_to_host(), debug_dict


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

    currentTime, verbose_1, verbose_2 = (time.perf_counter(),
                                         config.verbose >= 1,
                                         config.verbose >= 2)
    
    # reading image stack
    dng_stack = load_dng_burst(burst_path)
    ref_raw, raw_comp = dng_stack.get_raw_arrays() # Scale [black level, whiteleve] -> [0, 1] WITHOUT WB, clipping or anything

    if config.noise_model.alpha is not None:
        if config.noise_model.beta is None:
            raise ValueError("If alpha is provided, beta must also be provided.")
        # User provided custom values.
        print("Using user-provided alpha and beta values")
        config.noise_model.alpha, config.noise_model.beta = expand_noise_profile_to_rgbg(
            config.noise_model.alpha, config.noise_model.beta
        )
    else:
        # Extract alpha and beta from the DNG tags.
        config.noise_model.alpha = dng_stack.alpha
        config.noise_model.beta = dng_stack.beta

    if config.robustness.enabled and config.robustness.noise_correction:
        if config.noise_model.lut_path is None:
            raise ValueError(
                "robustness noise correction requires --noise-model.lut-path PATH; "
                "generate it with python -m handheld_super_resolution.monte_carlo"
            )
        noise_lut = load_noise_lut(
            config.noise_model.lut_path,
            expected_alpha=config.noise_model.alpha,
            expected_beta=config.noise_model.beta,
        )
        sigma_sq_curve = noise_lut.sigma_noise_sq
        d_sq_curve = noise_lut.d_noise_sq
    else:
        # The CUDA call signature remains fixed when correction is disabled.
        sigma_sq_curve = np.zeros(2, dtype=np.float32)
        d_sq_curve = np.zeros(2, dtype=np.float32)
    
    
    if verbose_2:   
        currentTime = getTime(currentTime, ' -- Read raw files')

    #### Estimating ref image SNR
    if config.force_snr is None:
        snr = estimate_image_snr(ref_raw, config.noise_model.alpha, config.noise_model.beta)
        if verbose_1:
            print(f"Estimated snr: {snr:.2f} dB")
    else:
        snr = config.force_snr
    
    update_snr_config(config, snr)
    
    sanitize_config(config, ref_raw.shape)

    if verbose_1:
        print(f"\nRuntime configuration:\n{config.dump()}\n")

    config.noise_model.sigma_sq_curve = sigma_sq_curve.tolist()
    config.noise_model.d_sq_curve = d_sq_curve.tolist()
    
    
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
        hr_output, debug_dict = main(ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE), raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE), config)

    if np.any(np.isnan(hr_output)):
        mask = np.any(np.isnan(hr_output), axis=-1)
        warnings.warn(f"Found {mask.sum()} pixels with NaNs in output")
        hr_output[mask] = np.array(NAN_COLOR, DEFAULT_NUMPY_FLOAT_TYPE)/255. if config.debug else [0., 0., 0.]

    if np.any(np.isinf(hr_output)):
        warnings.warn(f"Found {(np.isinf(hr_output)).sum()} Inf in output")

    #### Deflip the image
    hr_output = np.moveaxis(hr_output, -1, 0)
    hr_output = rggb_to_cfa(hr_output, dng_stack.cfa)
    hr_output = np.moveaxis(hr_output, 0, -1)
    if 'accumulated robustness' in debug_dict:
        debug_dict['accumulated robustness'] = rggb_to_cfa(debug_dict['accumulated robustness'], dng_stack.cfa)
    

    #### post processing
    post_processing_enabled = config.postprocessing.enabled
    
    if post_processing_enabled:
        if verbose_2:
            print('-- Post processing image')
        
        hr_output = raw2rgb.postprocess(hr_output, dng_stack, config) 
        
    if 'accumulated robustness' in debug_dict and config.postprocessing.orientate_image and 'Image Orientation' in dng_stack.tags:
            ori = dng_stack.tags['Image Orientation'].values[0]
            debug_dict['accumulated robustness'] = apply_orientation(debug_dict['accumulated robustness'], ori)
    
    
    
    return hr_output, debug_dict
