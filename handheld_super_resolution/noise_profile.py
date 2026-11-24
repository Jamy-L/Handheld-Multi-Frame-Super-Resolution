from pathlib import Path

import exifread


def expand_noise_profile_to_rgbg(
    alpha: list[float] | tuple[float, ...], beta: list[float] | tuple[float, ...]
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Expand one- or three-plane profiles to R, G1, B, G2 order."""
    assert (isinstance(alpha, list) and isinstance(beta, list)) or (
        isinstance(alpha, tuple) and isinstance(beta, tuple)
    ), f"got {type(alpha)}, {type(beta)}"
    if len(alpha) != len(beta):
        raise ValueError(
            f"alpha and beta must contain the same number of values, got "
            f"{len(alpha)} and {len(beta)}"
        )
    if len(alpha) == 1:
        alpha_ = alpha[0]
        beta_ = beta[0]
        return (alpha_, alpha_, alpha_, alpha_), (beta_, beta_, beta_, beta_)
    if len(alpha) == 3:
        return (alpha[0], alpha[1], alpha[2], alpha[1]), (
            beta[0],
            beta[1],
            beta[2],
            beta[1],
        )
    if len(alpha) == 4:
        return tuple(alpha), tuple(beta)
    raise ValueError(f"noise profiles must contain 1, 3, or 4 values, got {len(alpha)}")


def noise_profile_from_tags(
    tags,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Read the DNG NoiseProfile tag in R, G1, B, G2 color-plane order."""
    tag = tags.get("Image Tag 0xC761")
    if tag is None:
        raise ValueError("DNG does not contain a NoiseProfile (0xC761) tag")
    values = [float(value[0]) for value in tag.values]
    if len(values) % 2:
        raise ValueError(
            f"DNG NoiseProfile must contain alpha/beta pairs, got {values!r}"
        )
    return expand_noise_profile_to_rgbg(values[::2], values[1::2])


def read_dng_noise_profile(
    path: str | Path,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Read only the noise profile needed by the Monte Carlo CLI."""
    path = Path(path)
    with path.open("rb") as raw_file:
        tags = exifread.process_file(raw_file, details=False)
    return noise_profile_from_tags(tags)
