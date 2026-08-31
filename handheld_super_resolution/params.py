import warnings
from copy import deepcopy

import numpy as np
from typing import List, Literal, Optional, Union, Tuple

from .config import Config, SNRBasedFloat, SNR_BASED


def runtime_config(config: Config):
    """Return a mutable runtime copy, leaving the caller's config reusable."""
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of handheld_super_resolution.config.Config")
    return deepcopy(config)


def sanitize_config(config: Config, imshape: Tuple[int, int]):
    if config.mode == "grey" and config.grey_method != "FFT":
        raise NotImplementedError("Grey level images should be obtained with FFT")
        
    assert config.scale >= 1

    if not config.robustness.enabled and (config.accumulated_robustness_denoiser.median.enabled or
                                             config.accumulated_robustness_denoiser.gauss.enabled or
                                             config.accumulated_robustness_denoiser.merge.enabled):
        raise ValueError("Accumulated robustness denoiser cannot be enabled if robustness is disabled.")
    
    if not config.robustness.enabled and config.robustness.save_mask:
        raise ValueError("Robustness mask cannot be saved if robustness is disabled.")

    assert config.merging.kernel_type in ['steerable', 'iso'], f"Unknown kernel type {config.merging.kernel_type}"
    assert config.mode in ["bayer", 'grey'], f"Unknown mode {config.mode}"

    if sum([1 if x.enabled else 0 for x in [config.accumulated_robustness_denoiser.median,
                                            config.accumulated_robustness_denoiser.gauss,
                                            config.accumulated_robustness_denoiser.merge]]) > 1:
        raise ValueError("Only one accumulated robustness denoiser can be enabled at a time.")

    assert config.alignment.ica.n_iter > 0, "Number of ICA iterations should be positive."
    assert config.alignment.ica.sigma_blur >= 0, f"Invalid sigma blur {config.alignment.ica.sigma_blur}."

    assert len(imshape) == 2, f"Input image shape should be 2D, got {imshape}."

    ts = config.alignment.tile_size
    assert isinstance(ts, int), f"invalid tile size: {ts}"

    # Checking if image pyramid is possible
    padded_imshape_x = ts*(int(np.ceil(imshape[1]/ts)))
    padded_imshape_y = ts*(int(np.ceil(imshape[0]/ts)))
    
    lvl_imshape_y, lvl_imshape_x = padded_imshape_y, padded_imshape_x
    for lvl, (factor, ts) in enumerate(zip(config.alignment.factors, config.alignment.tile_sizes)):
        lvl_imshape_y, lvl_imshape_x = np.floor(lvl_imshape_y/factor), np.floor(lvl_imshape_x/factor)
        
        n_tiles_y = lvl_imshape_y/ts
        n_tiles_x = lvl_imshape_x/ts
        
        if n_tiles_y < 1 or n_tiles_x < 1:
            raise ValueError("Image of shape {} is incompatible with the given "\
                             "block matching tile sizes and factors : at level {}, "\
                             "coarse image of shape {} cannot be divided into "\
                             "tiles of size {}.".format(
                                 imshape, lvl,
                                 (lvl_imshape_y, lvl_imshape_x),
                                 ts))

    # Esnure that the flow upscaling mode is valid
    valid_upsample_modes = ['nearest', 'bilinear', 'bicubic']
    assert config.alignment.flow_upscale_mode in valid_upsample_modes, \
        f"Unknown flow upscaling mode {config.alignment.flow_upscale_mode}, " \
        f"should be one of {valid_upsample_modes}."
    
def update_snr_config(config: Config, snr: float):
    if snr < 6 or snr > 30:
        warnings.warn(f"Clipped snr {snr} between 6 and 30.")
    snr = float(np.clip(snr, 6, 30))
    
    if config.alignment.tile_size == SNR_BASED:
        if snr <= 14:
            ts = 64
        elif snr <= 22:
            ts = 32
        else:
            ts = 16
        config.alignment.tile_size = ts
        
    config.alignment.tile_sizes = [int(config.alignment.tile_size * s) for s in config.alignment.tile_size_factors]

    if config.merging.kernel.k_detail == SNR_BASED:
        config.merging.kernel.k_detail = lerp(snr, [6, 30], [0.33, 0.25])
    if config.merging.kernel.k_denoise == SNR_BASED:
        config.merging.kernel.k_denoise = lerp(snr, [6, 30], [5.0, 3.0])
    if config.merging.kernel.D_th == SNR_BASED:
        config.merging.kernel.D_th = lerp(snr, [6, 30], [0.81, 0.71])
    if config.merging.kernel.D_tr == SNR_BASED:
        config.merging.kernel.D_tr = lerp(snr, [6, 30], [1.24, 1])


def lerp(x, x_range, y_range):
    """
    Linearly interpolate a scalar value x from x_range -> y_range.

    Parameters
    ----------
    x : float or int
        Input value.
    x_range : tuple[float, float]
        (x_min, x_max) range.
    y_range : tuple[float, float]
        (y_min, y_max) range.

    Returns
    -------
    float
        Interpolated value in y_range.
    """
    x0, x1 = x_range
    y0, y1 = y_range

    assert x0 < x1
    assert y0 != y1

    # normalized t
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))

    return y0 + (y1 - y0) * t
