"""Generate sqrt-domain robustness noise LUTs from a DNG NoiseProfile.

Example:
    python -m handheld_super_resolution.monte_carlo \
        --dng frame.dng --output data/frame_noise.npz
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from .noise_lut import NoiseLut, save_noise_lut
from .utils_dng import expand_noise_profile_to_rgbg, read_dng_noise_profile


@njit(inline="always")
def _noisy_raw_value(brightness: float, alpha: float, beta: float) -> float:
    variance = alpha * brightness + beta
    if variance < 0.0:
        variance = 0.0
    value = brightness + math.sqrt(variance) * np.random.standard_normal()
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(inline="always")
def _channel_patch_stats(brightness: float, alpha: float, beta: float):
    mean = 0.0
    moment2 = 0.0
    for sample in range(9):
        value = math.sqrt(_noisy_raw_value(brightness, alpha, beta))
        delta = value - mean
        mean += delta / (sample + 1)
        moment2 += delta * (value - mean)
    return mean, max(moment2 / 9.0, 0.0)


@njit(inline="always")
def _green_patch_stats(
    brightness: float,
    alpha_g1: float,
    beta_g1: float,
    alpha_g2: float,
    beta_g2: float,
):
    mean = 0.0
    moment2 = 0.0
    for sample in range(9):
        green1 = _noisy_raw_value(brightness, alpha_g1, beta_g1)
        green2 = _noisy_raw_value(brightness, alpha_g2, beta_g2)
        value = math.sqrt(0.5 * (green1 + green2))
        delta = value - mean
        mean += delta / (sample + 1)
        moment2 += delta * (value - mean)
    return mean, max(moment2 / 9.0, 0.0)


@njit(parallel=True, cache=True)
def _simulate_chunk(
    alpha: np.ndarray,
    beta: np.ndarray,
    bin_count: int,
    global_start: int,
    global_stop: int,
    total_trials: int,
    seed: int,
    work_blocks: int,
):
    counts = np.zeros((work_blocks, bin_count), dtype=np.int64)
    sigma_sum = np.zeros((work_blocks, bin_count), dtype=np.float64)
    sigma_sum_sq = np.zeros((work_blocks, bin_count), dtype=np.float64)
    d_sum = np.zeros((work_blocks, bin_count), dtype=np.float64)
    d_sum_sq = np.zeros((work_blocks, bin_count), dtype=np.float64)
    chunk_trials = global_stop - global_start

    for block in prange(work_blocks):
        start = global_start + (chunk_trials * block) // work_blocks
        stop = global_start + (chunk_trials * (block + 1)) // work_blocks
        block_seed = (seed + 104729 * (block + 1) + 1000003 * global_start) & 0x7fffffff
        np.random.seed(block_seed)

        for trial in range(start, stop):
            # Midpoint stratification gives an exactly uniform latent calibration prior.
            latent = (trial + 0.5) / total_trials

            ref_r, ref_var_r = _channel_patch_stats(latent, alpha[0], beta[0])
            ref_g, ref_var_g = _green_patch_stats(
                latent, alpha[1], beta[1], alpha[3], beta[3]
            )
            ref_b, ref_var_b = _channel_patch_stats(latent, alpha[2], beta[2])

            mov_r, _ = _channel_patch_stats(latent, alpha[0], beta[0])
            mov_g, _ = _green_patch_stats(
                latent, alpha[1], beta[1], alpha[3], beta[3]
            )
            mov_b, _ = _channel_patch_stats(latent, alpha[2], beta[2])

            measured_brightness = (ref_r + ref_g + ref_b) / 3.0
            index = int(math.floor(measured_brightness * (bin_count - 1) + 0.5))
            if index < 0:
                index = 0
            elif index >= bin_count:
                index = bin_count - 1

            sigma_sq = ref_var_r + ref_var_g + ref_var_b
            delta_r = ref_r - mov_r
            delta_g = ref_g - mov_g
            delta_b = ref_b - mov_b
            d_sq = delta_r * delta_r + delta_g * delta_g + delta_b * delta_b

            counts[block, index] += 1
            sigma_sum[block, index] += sigma_sq
            sigma_sum_sq[block, index] += sigma_sq * sigma_sq
            d_sum[block, index] += d_sq
            d_sum_sq[block, index] += d_sq * d_sq

    return counts, sigma_sum, sigma_sum_sq, d_sum, d_sum_sq


def _interpolate_missing(values: np.ndarray, populated: np.ndarray) -> np.ndarray:
    indices = np.arange(values.size)
    if not np.any(populated):
        raise RuntimeError("Monte Carlo simulation did not populate any measured-brightness bins")
    return np.interp(indices, indices[populated], values[populated])


def _format_duration(seconds: float) -> str:
    """Format an ETA compactly enough to redraw on one terminal line."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _finalize_histograms(
    counts: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    populated = counts > 0
    means = np.zeros(counts.size, dtype=np.float64)
    sem = np.zeros(counts.size, dtype=np.float64)
    means[populated] = sums[populated] / counts[populated]
    variance = np.zeros(counts.size, dtype=np.float64)
    variance[populated] = np.maximum(
        sums_sq[populated] / counts[populated] - means[populated] ** 2,
        0.0,
    )
    sem[populated] = np.sqrt(variance[populated] / counts[populated])
    return _interpolate_missing(means, populated), _interpolate_missing(sem, populated)


def save_diagnostic_plot(npz_path: Path, lut: NoiseLut) -> Path:
    """Save brightness occupancy, d², and variance diagnostics beside a LUT."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot_path = Path(npz_path).with_suffix(".diagnostics.png")
    figure = Figure(figsize=(10, 11), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    brightness = np.asarray(lut.brightness, dtype=np.float64)

    axes[0].plot(brightness, lut.bin_counts, drawstyle="steps-mid", linewidth=1.0)
    axes[0].set_ylabel("MC trials / bin")
    axes[0].set_title("Monte Carlo population by measured reference brightness")

    d_sq = np.asarray(lut.d_noise_sq, dtype=np.float64)
    d_sq_sem = np.asarray(lut.d_noise_sq_sem, dtype=np.float64)
    axes[1].plot(brightness, d_sq, color="tab:orange", linewidth=1.5)
    axes[1].fill_between(
        brightness,
        np.maximum(d_sq - d_sq_sem, 0.0),
        d_sq + d_sq_sem,
        color="tab:orange",
        alpha=0.2,
        linewidth=0,
        label="±1 SEM",
    )
    axes[1].set_ylabel(r"$E[d^2\mid m_f]$")
    axes[1].set_title("Reference–moving noise distance")
    axes[1].legend(loc="best")

    sigma_sq = np.asarray(lut.sigma_noise_sq, dtype=np.float64)
    sigma_sq_sem = np.asarray(lut.sigma_noise_sq_sem, dtype=np.float64)
    axes[2].plot(brightness, sigma_sq, color="tab:green", linewidth=1.5)
    axes[2].fill_between(
        brightness,
        np.maximum(sigma_sq - sigma_sq_sem, 0.0),
        sigma_sq + sigma_sq_sem,
        color="tab:green", alpha=0.2, linewidth=0, label="±1 SEM",
    )
    axes[2].set_ylabel(r"$E[\sigma^2\mid m_f]$")
    axes[2].set_xlabel(r"Measured reference brightness $m_f$")
    axes[2].set_title("Reference-patch noise variance")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    figure.savefig(plot_path, dpi=160)
    return plot_path


def generate_noise_lut(
    alpha: Sequence[float],
    beta: Sequence[float],
    *,
    bins: int = 1001,
    trials: int = 10_000_000,
    seed: int = 0,
    threads: int = 0,
    chunk_size: int = 250_000,
    show_progress: bool = True,
) -> NoiseLut:
    """Simulate and bin squared nuisance statistics by measured brightness."""
    alpha, beta = expand_noise_profile_to_rgbg(alpha, beta)
    alpha_array = np.asarray(alpha, dtype=np.float64)
    beta_array = np.asarray(beta, dtype=np.float64)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if trials < 1:
        raise ValueError("trials must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if np.any(~np.isfinite(alpha_array)) or np.any(~np.isfinite(beta_array)):
        raise ValueError("alpha and beta must be finite")
    if np.any(alpha_array < 0.0) or np.any(beta_array < 0.0):
        raise ValueError("alpha and beta must be nonnegative")
    if threads < 0:
        raise ValueError("threads must be zero or positive")
    if threads:
        set_num_threads(threads)

    compile_start = time.perf_counter()
    _simulate_chunk(alpha_array, beta_array, 2, 0, 1, 1, seed, 1)
    compile_elapsed = time.perf_counter() - compile_start
    if show_progress:
        print(f"Numba ready in {compile_elapsed:.2f}s; using {get_num_threads()} threads", flush=True)

    counts = np.zeros(bins, dtype=np.int64)
    sigma_sum = np.zeros(bins, dtype=np.float64)
    sigma_sum_sq = np.zeros(bins, dtype=np.float64)
    d_sum = np.zeros(bins, dtype=np.float64)
    d_sum_sq = np.zeros(bins, dtype=np.float64)
    simulation_start = time.perf_counter()

    completed = 0
    while completed < trials:
        stop = min(completed + chunk_size, trials)
        block_count = max(1, min(get_num_threads() * 4, stop - completed))
        partial = _simulate_chunk(
            alpha_array, beta_array, bins, completed, stop, trials, seed, block_count
        )
        counts += np.sum(partial[0], axis=0)
        sigma_sum += np.sum(partial[1], axis=0)
        sigma_sum_sq += np.sum(partial[2], axis=0)
        d_sum += np.sum(partial[3], axis=0)
        d_sum_sq += np.sum(partial[4], axis=0)
        completed = stop

        if show_progress:
            elapsed = time.perf_counter() - simulation_start
            rate = completed / max(elapsed, 1e-12)
            eta = (trials - completed) / max(rate, 1e-12)
            print(
                f"\r{completed:,}/{trials:,} trials ({100.0 * completed / trials:5.1f}%)  "
                f"{rate:,.0f} trials/s  elapsed {_format_duration(elapsed)}  "
                f"ETA {_format_duration(eta)}",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    sigma_curve, sigma_sem = _finalize_histograms(counts, sigma_sum, sigma_sum_sq)
    d_curve, d_sem = _finalize_histograms(counts, d_sum, d_sum_sq)
    sparse = counts < 256
    if np.any(sparse):
        warnings.warn(
            f"{int(np.sum(sparse))}/{bins} measured-brightness bins contain fewer than "
            "256 trials; increase --trials for a smoother conditional estimate"
        )

    if show_progress:
        print(
            f"Bins: {int(np.sum(counts > 0))}/{bins} populated; "
            f"min/median count={int(np.min(counts))}/"
            f"{float(np.median(counts)):.0f}; "
            f"max SEM sigma^2={float(np.max(sigma_sem)):.3g}, "
            f"d^2={float(np.max(d_sem)):.3g}"
        )

    return NoiseLut(
        brightness=np.linspace(0.0, 1.0, bins, dtype=np.float32),
        sigma_noise_sq=sigma_curve.astype(np.float32),
        d_noise_sq=d_curve.astype(np.float32),
        bin_counts=counts,
        sigma_noise_sq_sem=sigma_sem.astype(np.float32),
        d_noise_sq_sem=d_sem.astype(np.float32),
        alpha=alpha_array,
        beta=beta_array,
    )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dng", type=Path, help="DNG whose NoiseProfile supplies alpha/beta")
    source.add_argument("--alpha", nargs="+", type=float, help="1, 3, or 4 alpha values")
    parser.add_argument("--beta", nargs="+", type=float, help="1, 3, or 4 beta values; required with --alpha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=1001)
    parser.add_argument("--trials", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="0 uses all Numba threads")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.dng is not None:
        if args.beta is not None:
            raise SystemExit("--beta may only be used together with --alpha")
        alpha, beta = read_dng_noise_profile(args.dng)
        source_dng = str(args.dng)
    else:
        if args.beta is None:
            raise SystemExit("--beta is required when --alpha is used")
        alpha, beta = expand_noise_profile_to_rgbg(args.alpha, args.beta)
        source_dng = ""

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; pass --overwrite to replace it")
    if args.output.suffix.lower() != ".npz":
        raise SystemExit("--output must end in .npz")

    print(f"alpha RGBG (R, G1, B, G2): {alpha}")
    print(f"beta  RGBG (R, G1, B, G2): {beta}")
    lut = generate_noise_lut(
        alpha,
        beta,
        bins=args.bins,
        trials=args.trials,
        seed=args.seed,
        threads=args.threads,
        chunk_size=args.chunk_size,
    )
    save_noise_lut(args.output, lut, trials=args.trials, seed=args.seed, source_dng=source_dng)
    print(f"Saved {args.output}")
    plot_path = save_diagnostic_plot(args.output, lut)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
