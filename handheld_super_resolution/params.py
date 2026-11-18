import warnings
from copy import deepcopy
from fractions import Fraction

import numpy as np

from .config import SNR_BASED, Config, GrayscaleMethod


def runtime_config(config: Config):
    """Return a mutable runtime copy, leaving the caller's config reusable."""
    if not isinstance(config, Config):
        raise TypeError("config must be an instance of handheld_super_resolution.config.Config")
    return deepcopy(config)


def sanitize_config(config: Config, imshape: tuple[int, int]):
    if config.scale < 1:
        raise ValueError("scale must be >= 1")

    if config.merging.kernel_type not in ("steerable", "iso"):
        raise ValueError(f"Unknown kernel type {config.merging.kernel_type}")
    if config.mode not in ("bayer", "grey"):
        raise ValueError(f"Unknown mode {config.mode}")
    if len(imshape) != 2:
        raise ValueError(f"Input image shape should be 2D, got {imshape}")

    alignment = config.alignment
    expected_tiles = alignment_tile_sizes(config)
    if alignment.tile_sizes != expected_tiles:
        raise ValueError("alignment.tile_sizes must be derived from tile_size and search_radii")

    height, width = imshape
    if height <= 0 or width <= 0:
        raise ValueError(f"Input image dimensions must be positive, got {imshape}")
    if config.mode == "bayer":
        if height % 2 or width % 2:
            raise ValueError("Bayer input must have even dimensions")
        if alignment.grey_method is GrayscaleMethod.DECIMATING:
            height, width = height // 2, width // 2
    for level, tile in enumerate(alignment.tile_sizes):
        if level:
            factor = alignment.factors[level - 1]
            height = (height + factor - 1) // factor
            width = (width + factor - 1) // factor
        if height < tile or width < tile:
            raise ValueError(
                f"Image of shape {imshape} is incompatible with alignment at level {level}: "
                f"image of shape {(height, width)} cannot fit tiles of size {tile}"
            )


def alignment_tile_sizes(config: Config) -> list[int]:
    """Derive fine-to-coarse tile sizes."""
    alignment = config.alignment
    levels = len(alignment.factors) + 1
    if len(alignment.search_radii) != levels or len(alignment.block_matching.metrics) != levels:
        raise ValueError("search_radii and block_matching.metrics must have len(factors) + 1 entries")
    if any(type(factor) is not int or factor < 2 or factor & (factor - 1)
           for factor in alignment.factors):
        raise ValueError("alignment.factors must be powers of two >= 2")
    if any(type(radius) is not int or radius < 0 for radius in alignment.search_radii):
        raise ValueError("alignment.search_radii must be nonnegative integers")
    if any(metric not in ("L1", "L2") for metric in alignment.block_matching.metrics):
        raise ValueError("block_matching.metrics entries must be L1 or L2")
    if type(alignment.ica.n_iter) is not int or alignment.ica.n_iter < 0:
        raise ValueError("alignment.ica.n_iter must be a nonnegative integer")
    if type(alignment.tile_size) is not int or alignment.tile_size not in (8, 16, 32, 64):
        raise ValueError("alignment.tile_size must be 8, 16, 32 or 64 after SNR selection")

    tiles = [alignment.tile_size]
    cap = Fraction(64)
    for level, radius in enumerate(alignment.search_radii[1:], start=1):
        tile = max(8, min(tiles[-1], cap))
        if tile != int(tile) or int(tile) & (int(tile) - 1):
            raise ValueError(
                f"alignment.search_radii produce tile size {tile} at level {level}; "
                "tile sizes must be powers of two"
            )
        tiles.append(int(tile))
        cap /= radius + 1
    return tiles


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
        
    config.alignment.tile_sizes = alignment_tile_sizes(config)

    if config.merging.kernel.sigma_detail == SNR_BASED:
        config.merging.kernel.sigma_detail = lerp(snr, [6, 30], [0.33, 0.25])
    if config.merging.kernel.k_denoise == SNR_BASED:
        config.merging.kernel.k_denoise = lerp(snr, [6, 30], [5.0, 3.0])
    if config.merging.kernel.D_th == SNR_BASED:
        config.merging.kernel.D_th = lerp(snr, [6, 30], [1e-2, 1e-3])
    if config.merging.kernel.D_tr == SNR_BASED:
        config.merging.kernel.D_tr = lerp(snr, [6, 30], [1e-2, 1e-3])


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
