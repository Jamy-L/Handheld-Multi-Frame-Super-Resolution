"""Typed configuration schema for handheld super-resolution."""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import List, Literal, Optional, Tuple, Union

import tyro


SNR_BASED = "SNR_based"
SNRBasedFloat = Union[float, Literal["SNR_based"]]
TileSize = Union[int, Literal["SNR_based"]]


def _format_config_fields(config, indent: int) -> List[str]:
    visible_fields = [config_field for config_field in fields(config) if config_field.repr]
    lines: List[str] = []

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
            lines.extend(
                f"{continuation_indent}{line}" for line in rendered_lines[1:]
            )

    return lines


@dataclass
class NoiseModelConfig:
    """Sensor noise model in EXIF R, G1, B, G2 plane order."""

    lut_path: Optional[Path] = None
    alpha: Optional[Tuple[float, float, float, float]] = None
    beta: Optional[Tuple[float, float, float, float]] = None
    sigma_sq_curve: tyro.conf.Suppress[List[float]] = field(default_factory=list, init=False, repr=False)
    d_sq_curve: tyro.conf.Suppress[List[float]] = field(default_factory=list, init=False, repr=False)


@dataclass
class BlockMatchingConfig:
    metrics: List[Literal["L1", "L2"]] = field(default_factory=lambda: ["L1", "L2", "L2", "L2"])


@dataclass
class ICAConfig:
    n_iter: int = 3
    sigma_blur: float = 0
    clip: bool = True


@dataclass
class AlignmentConfig:
    grey_method: Literal["FFT"] = "FFT"
    search_radii: List[int] = field(default_factory=lambda: [1, 4, 4, 4])
    flow_upscale_mode: Literal["nearest", "bilinear", "bicubic"] = "bilinear"
    factors: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    tile_size: TileSize = SNR_BASED
    tile_size_factors: List[float] = field(default_factory=lambda: [1, 1, 1, 0.5])
    tile_sizes: tyro.conf.Suppress[List[int]] = field(default_factory=list, init=False)
    block_matching: BlockMatchingConfig = field(default_factory=BlockMatchingConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)


@dataclass
class RobustnessConfig:
    enabled: bool = True
    noise_correction: bool = True
    save_mask: bool = True
    t: float = 0.12
    s1: float = 2
    s2: float = 12
    Mt: float = 0.8


@dataclass
class KernelConfig:
    k_detail: SNRBasedFloat = SNR_BASED
    k_denoise: SNRBasedFloat = SNR_BASED
    D_th: SNRBasedFloat = SNR_BASED
    D_tr: SNRBasedFloat = SNR_BASED
    k_stretch: float = 4
    k_shrink: float = 2


@dataclass
class MergingConfig:
    kernel_type: Literal["steerable", "iso"] = "steerable"
    selection_law: Literal["hard_threshold", "linear"] = "linear"
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

    scale: float = 1
    mode: Literal["bayer", "grey"] = "bayer"
    debug: bool = False
    verbose: int = 1
    use_gat: bool = True
    noise_model: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    def dump(self) -> str:
        """Return a readable snapshot of the complete runtime configuration."""
        return "\n".join([f"{type(self).__name__}:", *_format_config_fields(self, 2)])
