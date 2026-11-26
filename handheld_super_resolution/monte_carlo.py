import argparse
from pathlib import Path

import numpy as np

from .noise_profile import expand_noise_profile_to_rgbg, read_dng_noise_profile


def generate_noise_lut(alpha, beta, *, points=10, samples_per_point=4096, seed=0):
    if points < 2 or samples_per_point < 1 or seed < 0:
        raise ValueError("require points >= 2, samples per point >= 1 and seed >= 0")
    alpha, beta = np.asarray(expand_noise_profile_to_rgbg(alpha, beta))
    if not all(np.isfinite(v).all() and (v >= 0).all() for v in (alpha, beta)):
        raise ValueError("noise coefficients must be finite and nonnegative")
    grid = np.linspace(0., 1., points)
    texture = np.zeros((2, points, 4), dtype=np.float32)
    weights = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]) / 16
    # Every column uses the same Gaussian sequence.
    rng = np.random.Generator(np.random.MT19937(seed))
    draws_1 = rng.standard_normal((samples_per_point, 9, 4))
    draws_2 = rng.standard_normal(draws_1.shape)
    for i, u in enumerate(grid):
        mean = u * u
        sigma = np.sqrt(beta + alpha * mean)
        raw_1 = mean + draws_1 * sigma
        raw_2 = mean + draws_2 * sigma
        r, g1, b, g2 = np.moveaxis(np.sqrt(np.maximum(raw_1, 0)), -1, 0)
        y_1 = (r + g1 + g2 + b) / 4
        y_2 = np.sqrt(np.maximum(raw_2, 0)).mean(axis=-1)
        y_error = np.mean((y_1 @ weights - u)**2)
        c1 = ((-r + g1 + g2 - b) / 4) @ weights
        c2 = (r / 2 - b / 4) @ weights
        texture[0, i] = (y_error, np.mean(c1**2), np.mean(c2**2),
                         np.sqrt(y_error + y_1.var(axis=1).mean()))
        texture[1, i, 0] = np.sqrt(np.mean(((g1 - g2)**2) @ weights))
        texture[1, i, 1] = np.mean((y_1 - y_2)**2)
    return grid, texture


def save_diagnostic_plot(path, grid, texture):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 7), layout="constrained")
    for row, (axis, labels) in enumerate(zip(figure.subplots(2), (
        ("Y mean square error", "C1 mean square", "C2 mean square", "Local std floor"),
        ("Green difference RMS", "Luminance difference mean square"),
    ))):
        for channel, label in enumerate(labels):
            axis.plot(grid**2, texture[row, :, channel], ".-", label=label)
        axis.set_yscale("log")
        axis.grid(True, which="major", alpha=0.4)
        axis.grid(True, which="minor", alpha=0.15)
        axis.set_xlabel("Linear RAW brightness (u²)")
        axis.legend()
    figure.savefig(path.with_suffix(".diagnostics.png"))


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dng", type=Path, help="DNG whose NoiseProfile supplies alpha/beta")
    source.add_argument("--alpha", nargs="+", type=float, help="1, 3, or 4 alpha values")
    parser.add_argument("--beta", nargs="+", type=float, help="1, 3, or 4 beta values; required with --alpha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--samples-per-point", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
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
    grid, texture = generate_noise_lut(alpha, beta, points=args.points,
                                      samples_per_point=args.samples_per_point, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, format_version=2, texture=texture, brightness=grid,
                        alpha=alpha, beta=beta, samples_per_point=args.samples_per_point,
                        seed=args.seed, source_dng=source_dng)
    save_diagnostic_plot(args.output, grid, texture)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
