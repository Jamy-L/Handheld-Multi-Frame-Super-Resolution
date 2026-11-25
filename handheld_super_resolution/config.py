"""Typed configuration schema for handheld super-resolution."""

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from fractions import Fraction
from pathlib import Path
from pprint import pformat
from typing import Annotated, Literal

import numpy as np
import tyro

SNR_BASED = "SNR_based"
SNRBasedFloat = float | Literal["SNR_based"]
TileSize = int | Literal["SNR_based"]


class KernelType(StrEnum):
    STEERABLE = "steerable"
    ISO = "iso"


class GrayscaleMethod(Enum):
    """Methods for converting a raw image to grayscale."""

    DEMOSAICING = "demosaicing"
    DECIMATING = "decimating"


def _format_config_fields(config, indent: int) -> list[str]:
    visible_fields = [
        config_field for config_field in fields(config) if config_field.repr
    ]
    lines: list[str] = []

    for config_field in visible_fields:
        value = getattr(config, config_field.name)
        prefix = " " * indent + f"{config_field.name}:"

        if is_dataclass(value):
            lines.append(prefix)
            lines.extend(_format_config_fields(value, indent + 2))
        else:
            width = max(20, 100 - len(prefix) - 1)
            rendered_lines = pformat(
                value, width=width, compact=True, sort_dicts=False
            ).splitlines()
            lines.append(f"{prefix} {rendered_lines[0]}")
            continuation_indent = " " * (len(prefix) + 1)
            lines.extend(f"{continuation_indent}{line}" for line in rendered_lines[1:])

    return lines


@dataclass
class NoiseModelConfig:
    """Sensor noise model in EXIF R, G1, B, G2 plane order."""

    lut_path: Path | None = None
    alpha: tuple[float, float, float, float] | None = None
    beta: tuple[float, float, float, float] | None = None


@dataclass
class BlockMatchingConfig:
    """Fine to coarse."""

    metrics: list[Literal["L1", "L2"]] = field(
        default_factory=lambda: ["L1", "L2", "L2", "L2", "L2"]
    )


@dataclass
class ICAConfig:
    n_iter: int = 3


@dataclass
class AlignmentConfig:
    block_matching: BlockMatchingConfig = field(default_factory=BlockMatchingConfig)
    # Fine to coarse
    search_radii: list[int] = field(default_factory=lambda: [1, 3, 3, 3, 3])
    factors: list[int] = field(default_factory=lambda: [2, 4, 4, 4])

    ica: ICAConfig = field(default_factory=ICAConfig)

    # How to compute raw -> grayscale
    grey_method: Literal[GrayscaleMethod.DEMOSAICING, GrayscaleMethod.DECIMATING] = (
        GrayscaleMethod.DECIMATING
    )
    tile_size: TileSize = SNR_BASED
    # Derived from tile_size and search_radii using the GCam 8.4 shrinking cap.
    tile_sizes: tyro.conf.Suppress[list[int]] = field(default_factory=list, init=False)

    def resolve_snr(self, snr: float) -> None:
        if self.tile_size == SNR_BASED:
            if snr < 0.5:
                self.tile_size = 32
            elif snr < 2.7:
                self.tile_size = 64
            else:
                self.tile_size = 16
        self.tile_sizes = self.derive_tile_sizes()

    def derive_tile_sizes(self) -> list[int]:
        """Derive fine-to-coarse tile sizes."""
        levels = len(self.factors) + 1
        if (
            len(self.search_radii) != levels
            or len(self.block_matching.metrics) != levels
        ):
            raise ValueError(
                "search_radii and block_matching.metrics must have len(factors) + 1 entries"
            )
        if any(
            type(factor) is not int or factor < 2 or factor & (factor - 1)
            for factor in self.factors
        ):
            raise ValueError("alignment.factors must be powers of two >= 2")
        if any(type(radius) is not int or radius < 0 for radius in self.search_radii):
            raise ValueError("alignment.search_radii must be nonnegative integers")
        if any(metric not in ("L1", "L2") for metric in self.block_matching.metrics):
            raise ValueError("block_matching.metrics entries must be L1 or L2")
        if type(self.ica.n_iter) is not int or self.ica.n_iter < 0:
            raise ValueError("alignment.ica.n_iter must be a nonnegative integer")
        if type(self.tile_size) is not int or self.tile_size not in (8, 16, 32, 64):
            raise ValueError(
                "alignment.tile_size must be 8, 16, 32 or 64 after SNR selection"
            )

        tiles = [self.tile_size]
        cap = Fraction(64)
        for level, radius in enumerate(self.search_radii[1:], start=1):
            tile = max(8, min(tiles[-1], cap))
            if tile != int(tile) or int(tile) & (int(tile) - 1):
                raise ValueError(
                    f"alignment.search_radii produce tile size {tile} at level {level}; "
                    "tile sizes must be powers of two"
                )
            tiles.append(int(tile))
            cap /= radius + 1
        return tiles


@dataclass
class RobustnessConfig:
    enabled: bool = True
    noise_correction: bool = True
    save_mask: bool = True
    # Scales green disagreement in the guide-color blend. 0 to disable.
    k_green_diff_impact: float = 2.0
    R_mult: float = 1.0  # Multiplier on normalized color distance.
    R_threshold: float = 0.0  # Offset in the rejection weight's base-2 exponent.
    M_th: float = 1e-4  # Minimum local flow variation for boosted rejection.
    k_std: float = (
        5.0  # Minimum local standard deviation in multiples of noise std for boosting.
    )
    R_boost: float = 6.0  # Extra rejection multiplier when both thresholds are met.


# Each curve clamps independently at its first and last SNR.
_KERNEL_SNR_KNOTS = {
    "sigma_detail": ((8, 0.33), (16, 0.25)),
    "k_denoise": ((6, 5.0), (12, 4.0), (30, 3.0)),
    "D_th": ((1, 0.01), (4, 0.002), (14, 0.001), (30, 0.001)),
    "D_tr": ((1, 0.02), (3, 0.015), (12, 0.009), (30, 0.006)),
    "k_stretch": ((16, 4.0),),
    "k_shrink": ((16, 2.2),),
}


@dataclass
class KernelConfig:
    enable_green_diff: bool = (
        True  # Use Bayer green disagreement as a floor on steerable kernel strength.
    )
    noise_correction: bool = True
    sigma_detail: SNRBasedFloat = SNR_BASED  # Smallest sigma
    k_denoise: SNRBasedFloat = SNR_BASED  # Multiplier to enlarge sigma
    k_stretch: SNRBasedFloat = SNR_BASED  # Stretches sigma_detail along edges
    k_shrink: SNRBasedFloat = SNR_BASED  # Shrinks sigma_detail across edges
    D_th: SNRBasedFloat = SNR_BASED
    D_tr: SNRBasedFloat = SNR_BASED

    def resolve_snr(self, snr: float) -> None:
        for name, knots in _KERNEL_SNR_KNOTS.items():
            if getattr(self, name) == SNR_BASED:
                signals, values = zip(*knots)
                setattr(self, name, float(np.interp(snr, signals, values)))


@dataclass
class MergingConfig:
    kernel_type: Annotated[KernelType, tyro.conf.EnumChoicesFromValues] = (
        KernelType.STEERABLE
    )  # isotropic mode uses sigma_detail
    kernel: KernelConfig = field(default_factory=KernelConfig)


@dataclass
class SharpeningConfig:
    enabled: bool = True
    amount: float = 1.5
    radius: float = 3


@dataclass
class PostprocessingConfig:
    enabled: bool = True
    do_white_balance: bool = True
    do_camera_to_linear_srgb: bool = True
    do_srgb_encoding: bool = True
    orientate_image: bool = True
    sharpening: SharpeningConfig = field(default_factory=SharpeningConfig)


@dataclass
class Config:
    """Configuration for the handheld multi-frame super-resolution pipeline."""

    scale: float = 2  # 1 for demosaicking
    debug: bool = False  # Dumps intermediate buffers as .png, and colours eventual NaN pixel in magenta
    verbose: int = 1
    force_snr: float | None = (
        None  # Override the estimated linear signal-to-noise ratio
    )
    noise_model: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    backend: str = "automatic"
    """SlangPy backend name. Use --list-devices to see available choices."""
    device: int | None = None
    """Zero-based adapter index on the selected backend (default: let SlangPy choose)."""

    @property
    def requires_noise_lut(self) -> bool:
        return self.robustness.noise_correction or self.merging.kernel.noise_correction

    def dump(self) -> str:
        """Return a readable snapshot of the complete runtime configuration."""
        return "\n".join([f"{type(self).__name__}:", *_format_config_fields(self, 2)])

    def sanitize(self) -> None:
        """Check user settings before loading a burst or resolving automatic values."""
        noise = self.noise_model
        if (noise.alpha is None) != (noise.beta is None):
            raise ValueError(
                "Both noise_model.alpha and noise_model.beta must be provided together"
            )
        if self.requires_noise_lut and noise.lut_path is None:
            raise ValueError(
                "enabled noise correction requires --noise-model.lut-path PATH"
            )
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if self.backend == "cpu":
            raise NotImplementedError(
                "CPU backend is not supported"
            )  # CPU backend doesnt support textures and samplers

        self.merging.kernel_type = KernelType(self.merging.kernel_type)

    def validate_alignment_shape(self, imshape: tuple[int, int]) -> None:
        """Check that resolved alignment tiles fit each pyramid level."""
        if len(imshape) != 2:
            raise ValueError(f"Input image shape should be 2D, got {imshape}")

        alignment = self.alignment
        expected_tiles = alignment.derive_tile_sizes()
        if alignment.tile_sizes != expected_tiles:
            raise ValueError(
                "alignment.tile_sizes must be derived from tile_size and search_radii"
            )

        height, width = imshape
        if height <= 0 or width <= 0:
            raise ValueError(f"Input image dimensions must be positive, got {imshape}")
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

    def resolve_snr(self, snr: float) -> None:
        self.alignment.resolve_snr(snr)
        self.merging.kernel.resolve_snr(snr)
