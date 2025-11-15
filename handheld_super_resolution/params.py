import warnings
import numpy as np

def sanitize_config(config, imshape):
    if config.mode == "grey" and config.grey_method != "FFT":
        raise NotImplementedError("Grey level images should be obtained with FFT")
        
    assert config.scale >= 1

    if not config.robustness.enabled and (config.accumulated_robustness_denoiser.median.enabled or
                                             config.accumulated_robustness_denoiser.gauss.enabled or
                                             config.accumulated_robustness_denoiser.merge.enabled):
        raise ValueError("Accumulated robustness denoiser cannot be enabled if robustness is disabled.")
    
    if not config.robustness.enabled and config.robustness.save_mask:
        raise ValueError("Robustness mask cannot be saved if robustness is disabled.")

    assert config.merging.kernel in ['steerable', 'iso'], f"Unknown kernel type {config.merging.kernel}"
    assert config.mode in ["bayer", 'grey'], f"Unknown mode {config.mode}"

    if sum([1 if x.enabled else 0 for x in [config.accumulated_robustness_denoiser.median,
                                            config.accumulated_robustness_denoiser.gauss,
                                            config.accumulated_robustness_denoiser.merge]]) > 1:
        raise ValueError("Only one accumulated robustness denoiser can be enabled at a time.")

    assert config.ica.tuning.n_iter > 0, "Number of ICA iterations should be positive."
    assert config.ica.tuning.sigma_blur >= 0, f"Invalid sigma blur {config.ica.tuning.sigma_blur}."

    assert len(imshape) == 2, f"Input image shape should be 2D, got {imshape}."

    Ts = config.block_matching.tuning.tile_size

    # Checking if block matching is possible
    padded_imshape_x = Ts*(int(np.ceil(imshape[1]/Ts)))
    padded_imshape_y = Ts*(int(np.ceil(imshape[0]/Ts)))
    
    lvl_imshape_y, lvl_imshape_x = padded_imshape_y, padded_imshape_x
    for lvl, (factor, ts) in enumerate(zip(config.block_matching.tuning.factors, config.block_matching.tuning.tile_sizes)):
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
    
def update_snr_config(config, SNR):
    SNR = np.clip(SNR, 6, 30)
    SNR = float(SNR)
    if SNR <= 14:
        Ts = 64
    elif SNR <= 22:
        Ts = 32
    else:
        Ts = 16
    
    if config.block_matching.tuning.tile_size != "SNR_based":
        assert isinstance(config.block_matching.tuning.tile_size, int), "tile_size should be an integer or 'SNR_based'"
        Ts = config.block_matching.tuning.tile_size
    else:
        config.block_matching.tuning.tile_size = Ts
        
    sizes = [int(Ts * s) for s in config.block_matching.tuning.tile_size_factors]
    config.block_matching.tuning.tile_sizes = sizes

    if config.merging.tuning.k_detail == "SNR_based":
        config.merging.tuning.k_detail = lerp(SNR, [6, 30], [0.33, 0.25])
    else:
        assert isinstance(config.merging.tuning.k_detail, float), "k_detail should be a float or 'SNR_based'"
    if config.merging.tuning.k_denoise == "SNR_based":
        config.merging.tuning.k_denoise = lerp(SNR, [6, 30], [5.0, 3.0])
    else:
        assert isinstance(config.merging.tuning.k_denoise, float), "k_denoise should be a float or 'SNR_based'"
    if config.merging.tuning.D_th == "SNR_based":
        config.merging.tuning.D_th = lerp(SNR, [6, 30], [0.81, 0.71])
    else:
        assert isinstance(config.merging.tuning.D_th, float), "D_th should be a float or 'SNR_based'"
    if config.merging.tuning.D_tr == "SNR_based":
        config.merging.tuning.D_tr = lerp(SNR, [6, 30], [1.24, 1])
    else:
        assert isinstance(config.merging.tuning.D_tr, float), "D_tr should be a float or 'SNR_based'"


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