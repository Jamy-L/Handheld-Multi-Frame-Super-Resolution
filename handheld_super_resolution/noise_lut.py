"""Load the packed SABRE texture and its endpoint-to-texel mapping."""

from pathlib import Path

import numpy as np
import slangpy as spy

from .noise_profile import expand_noise_profile_to_rgbg


def create_zero_noise_lut(device):
    return _upload_noise_lut(device, np.zeros((2, 10, 4), dtype=np.float32))


def load_noise_lut(device, path: Path, expected_alpha=None, expected_beta=None):
    with np.load(path, allow_pickle=False) as archive:
        if archive["format_version"].item() != 2:
            raise ValueError("expected a version 2 SABRE LUT; regenerate with monte_carlo")
        grid = archive["brightness"]
        points = grid.size
        limit = device.info.limits.max_texture_dimension_2d
        if points > limit:
            raise ValueError(f"noise LUT has {points} points; device 2D texture limit is {limit}")
        if grid.ndim != 1 or points < 2 or not np.allclose(
            grid, np.linspace(0, 1, points), rtol=0, atol=2e-7
        ):
            raise ValueError("noise LUT requires a uniform signal grid from 0 to 1")
        data = archive["texture"]
        if data.shape != (2, points, 4) or data.dtype != np.float32:
            raise ValueError(f"noise LUT texture must be float32 with shape (2, {points}, 4)")
        if not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("noise LUT statistics must be finite and nonnegative")
        if (expected_alpha is None) != (expected_beta is None):
            raise ValueError("provide both expected noise coefficients together")
        if expected_alpha is not None:
            profile = expand_noise_profile_to_rgbg(expected_alpha, expected_beta)
            for key, expected in zip(("alpha", "beta"), profile):
                values = archive[key]
                if values.shape != (4,) or not np.allclose(values, expected, rtol=1e-6, atol=1e-12):
                    raise ValueError(f"noise LUT {key} does not match the burst")

    return _upload_noise_lut(device, data)


def _upload_noise_lut(device, data):
    points = data.shape[1]
    texture = device.create_texture(
        width=points, height=2, mip_count=1, format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource, data=np.ascontiguousarray(data),
        sampler=device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear, mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        ),
        label="noise_estimates",
    )
    # Flow convention: uv = ((signal, row) - offset) * scale.
    # Use row = 0 or 1 only. Signal between 0 and 1. sampling signal = 0 or 1 will return the exact curve extremity.
    offset = (-0.5 / (points - 1), -0.5)
    scale = ((points - 1) / points, 0.5)
    return texture, offset, scale
