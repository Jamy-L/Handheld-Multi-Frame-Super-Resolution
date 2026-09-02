"""Versioned storage and validation for robustness noise-correction LUTs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .utils_dng import expand_noise_profile_to_rgbg


FORMAT_VERSION = 1
TRANSFORM_NAME = "clip_raw_then_sqrt_bayer_quad_rgb_v1"
LATENT_PRIOR = "uniform_stratified_0_1"


@dataclass(frozen=True)
class NoiseLut:
    """Noise curves and their originating RGBG-ordered sensor profile."""
    brightness: NDArray[np.float32]
    sigma_noise_sq: NDArray[np.float32]
    d_noise_sq: NDArray[np.float32]
    bin_counts: NDArray[np.int64]
    sigma_noise_sq_sem: NDArray[np.float32]
    d_noise_sq_sem: NDArray[np.float32]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]


def _scalar(data, key):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"noise LUT field {key!r} must be scalar")
    return value.reshape(()).item()


def load_noise_lut(
    path: Union[str, Path],
    expected_alpha=None,
    expected_beta=None,
) -> NoiseLut:
    """Load a LUT, rejecting stale transforms and mismatched noise profiles."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"noise LUT does not exist: {path}")

    required = {
        "format_version", "transform", "brightness", "sigma_noise_sq",
        "d_noise_sq", "bin_counts", "sigma_noise_sq_sem", "d_noise_sq_sem",
        "alpha_rgbg", "beta_rgbg", "patch_size", "latent_prior", "bins",
        "trials", "seed",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"noise LUT is missing fields: {sorted(missing)}")
            if int(_scalar(data, "format_version")) != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported noise LUT format version: {_scalar(data, 'format_version')}"
                )
            if str(_scalar(data, "transform")) != TRANSFORM_NAME:
                raise ValueError(f"noise LUT was generated for the wrong transform: {_scalar(data, 'transform')!r}")
            if int(_scalar(data, "patch_size")) != 3:
                raise ValueError("noise LUT must use 3x3 patches")
            stored_bins = int(_scalar(data, "bins"))
            stored_trials = int(_scalar(data, "trials"))
            stored_seed = int(_scalar(data, "seed"))
            if str(_scalar(data, "latent_prior")) != LATENT_PRIOR:
                raise ValueError(f"unsupported latent prior: {_scalar(data, 'latent_prior')!r}")

            brightness = np.asarray(data["brightness"], dtype=np.float32)
            sigma_sq = np.asarray(data["sigma_noise_sq"], dtype=np.float32)
            d_sq = np.asarray(data["d_noise_sq"], dtype=np.float32)
            raw_counts = np.asarray(data["bin_counts"])
            if not np.issubdtype(raw_counts.dtype, np.integer):
                raise ValueError("noise LUT bin counts must use an integer dtype")
            counts = raw_counts.astype(np.int64, copy=False)
            sigma_sem = np.asarray(data["sigma_noise_sq_sem"], dtype=np.float32)
            d_sem = np.asarray(data["d_noise_sq_sem"], dtype=np.float32)
            alpha = np.asarray(data["alpha_rgbg"], dtype=np.float64)
            beta = np.asarray(data["beta_rgbg"], dtype=np.float64)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(("noise LUT", "unsupported")):
            raise
        raise ValueError(f"cannot read noise LUT {path}: {error}") from error

    arrays = (brightness, sigma_sq, d_sq, counts, sigma_sem, d_sem)
    if brightness.ndim != 1 or brightness.size < 2:
        raise ValueError("noise LUT brightness must be a one-dimensional array with at least two bins")
    if stored_bins != brightness.size:
        raise ValueError(
            f"noise LUT bins metadata ({stored_bins}) does not match its curves ({brightness.size})"
        )
    if any(array.shape != brightness.shape for array in arrays[1:]):
        raise ValueError("all noise LUT curves must have the same one-dimensional shape")
    expected_grid = np.linspace(0.0, 1.0, brightness.size, dtype=np.float32)
    if not np.allclose(brightness, expected_grid, rtol=0.0, atol=2e-7):
        raise ValueError("noise LUT brightness must be a uniform measured-brightness grid from 0 to 1")
    if alpha.shape != (4,) or beta.shape != (4,):
        raise ValueError("noise LUT alpha and beta must be RGBG vectors of length four")
    if not all(np.all(np.isfinite(array)) for array in (brightness, sigma_sq, d_sq, sigma_sem, d_sem, alpha, beta)):
        raise ValueError("noise LUT contains non-finite values")
    if np.any(sigma_sq < 0) or np.any(d_sq < 0) or np.any(sigma_sem < 0) or np.any(d_sem < 0):
        raise ValueError("noise LUT squared statistics and standard errors must be nonnegative")
    if np.any(alpha < 0) or np.any(beta < 0):
        raise ValueError("noise LUT alpha and beta must be nonnegative")
    if np.any(counts < 0):
        raise ValueError("noise LUT bin counts must be nonnegative")
    if stored_trials < 1 or stored_seed < 0:
        raise ValueError("noise LUT trials must be positive and seed must be nonnegative")
    if int(np.sum(counts)) != stored_trials:
        raise ValueError(
            f"noise LUT bin counts sum to {int(np.sum(counts))}, expected {stored_trials} trials"
        )

    if (expected_alpha is None) != (expected_beta is None):
        raise ValueError("expected_alpha and expected_beta must be provided together")
    if expected_alpha is not None:
        expanded_alpha, expanded_beta = expand_noise_profile_to_rgbg(expected_alpha, expected_beta)
        if not np.allclose(alpha, expanded_alpha, rtol=1e-6, atol=1e-12):
            raise ValueError(f"noise LUT alpha does not match the burst: LUT={alpha}, burst={expanded_alpha}")
        if not np.allclose(beta, expanded_beta, rtol=1e-6, atol=1e-12):
            raise ValueError(f"noise LUT beta does not match the burst: LUT={beta}, burst={expanded_beta}")

    return NoiseLut(brightness, sigma_sq, d_sq, counts, sigma_sem, d_sem, alpha, beta)


def save_noise_lut(
    path: Union[str, Path],
    lut: NoiseLut,
    *,
    trials: int,
    seed: int,
    source_dng: str = "",
) -> None:
    """Save a generated LUT without requiring pickle to read its metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.int32(FORMAT_VERSION),
        transform=np.asarray(TRANSFORM_NAME),
        latent_prior=np.asarray(LATENT_PRIOR),
        brightness=lut.brightness,
        sigma_noise_sq=lut.sigma_noise_sq,
        d_noise_sq=lut.d_noise_sq,
        bin_counts=lut.bin_counts,
        sigma_noise_sq_sem=lut.sigma_noise_sq_sem,
        d_noise_sq_sem=lut.d_noise_sq_sem,
        alpha_rgbg=lut.alpha,
        beta_rgbg=lut.beta,
        patch_size=np.int32(3),
        bins=np.int32(lut.brightness.size),
        trials=np.int64(trials),
        seed=np.int64(seed),
        source_dng=np.asarray(source_dng),
    )
