import cv2
import numpy as np
import slangpy as spy

from .config import Config
from .debug_writer import DebugWriter
from .utils import DEFAULT_NUMPY_FLOAT_TYPE
from .utils_image import decimate_2x2

TS = 8


def _tile_stats(
    image: np.ndarray, tile_size: int, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    cropped = image[: rows * tile_size, : cols * tile_size]
    tiles = cropped.reshape(rows, tile_size, cols, tile_size)
    return tiles.mean(axis=(1, 3)), tiles.std(axis=(1, 3))


def compute_unblocker_texture(
    ref_raw: np.ndarray, config: Config, debug_writer: DebugWriter | None = None
) -> tuple[spy.Texture, spy.Sampler, tuple[float, float]]:
    """Upload the score grid and return its texture, sampler, and UV scale."""
    height, width = ref_raw.shape
    rows, cols = height // TS, width // TS
    if config.unblocker.enabled and rows and cols:
        score = _compute_scores(ref_raw, config, debug_writer, rows, cols)
    else:
        score = np.zeros((max(1, rows), max(1, cols)), dtype=DEFAULT_NUMPY_FLOAT_TYPE)

    if debug_writer is not None:
        upper = score[np.isfinite(score)].max(initial=0.0)
        debug_writer.write_scalar(
            "unblocker", score, (0.0, float(upper) or 1.0), cv2.COLORMAP_MAGMA
        )

    # Each texel is centered on an 8x8 tile. Clamp sampling over cropped edges.
    uv_scale = (width / (TS * score.shape[1]), height / (TS * score.shape[0]))
    device = spy.current_device()
    sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        address_u=spy.TextureAddressingMode.clamp_to_edge,
        address_v=spy.TextureAddressingMode.clamp_to_edge,
    )
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        width=score.shape[1],
        height=score.shape[0],
        format=spy.Format.r32_float,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource,
        data=score,
        sampler=sampler,
        label="unblocker",
    )
    return texture, sampler, uv_scale


def _compute_scores(
    ref_raw: np.ndarray,
    config: Config,
    debug_writer: DebugWriter | None,
    rows: int,
    cols: int,
) -> np.ndarray:
    ref_raw = np.asarray(ref_raw, dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    # Assume RGGB format
    rggb = np.stack(
        (ref_raw[::2, ::2], ref_raw[::2, 1::2], ref_raw[1::2, ::2], ref_raw[1::2, ::2])
    )
    # Compute Green frame
    padded = np.pad(rggb, ((0, 0), (1, 1), (1, 1)), mode="edge")
    green = rggb.copy()
    green[0] = (
        (  # Interpolate greens at red sites.
            padded[1, 1:-1, :-2]  # left: G1 from left quad
            + padded[2, :-2, 1:-1]  # Top G2 from top quad
            + padded[1, 1:-1, 1:-1]  # right: G1 from current quad
            + padded[2, 1:-1, 1:-1]  # bottom: G2 from current quad
        )
        / 4
    )
    green[3] = (
        (  # Interpolate greens at blue sites.
            padded[2, 1:-1, 1:-1]  # left: G2 from current quad
            + padded[1, 1:-1, 1:-1]  # Top G1 from current quad
            + padded[2, 1:-1, 2:]  # right: G2 from right quad
            + padded[1, 2:, 1:-1]  # bottom: G1 from bottom quad
        )
        / 4
    )
    green_frame = np.empty_like(ref_raw)
    green_frame[::2, ::2] = green[0]
    green_frame[::2, 1::2] = green[1]
    green_frame[1::2, ::2] = green[2]
    green_frame[1::2, 1::2] = green[3]
    if debug_writer is not None:
        debug_writer.write_rgb("unblocker_hr_green", green_frame)

    lr = decimate_2x2(green_frame)
    hr_mean, hr_std = _tile_stats(green_frame, TS, rows, cols)
    _, coarse = _tile_stats(lr, TS // 2, rows, cols)
    alpha = config.noise_model.alpha
    beta = config.noise_model.beta
    if alpha is not None and beta is not None:
        # EXIF plane order is R, G1, B, G2.
        green_alpha = (alpha[1] + alpha[3]) * 0.5
        green_beta = (beta[1] + beta[3]) * 0.5
        noise_std = np.sqrt(green_beta + hr_mean * green_alpha)
        hr_std = (hr_std - noise_std).clip(min=0)

    fine = hr_std
    score = np.zeros_like(fine)
    has_coarse_texture = coarse != 0
    ratio = fine[has_coarse_texture] / coarse[has_coarse_texture]
    unblocker = config.unblocker
    score[has_coarse_texture] = np.clip(
        unblocker.fallback_value * ratio / unblocker.s - unblocker.t, 0, 1
    )
    score[~has_coarse_texture & (fine > 0)] = unblocker.fallback_value

    return score
