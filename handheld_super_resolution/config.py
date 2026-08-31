"""Typed configuration schema for handheld super-resolution."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import tyro


SNR_BASED = "SNR_based"
SNRBasedFloat = Union[float, Literal["SNR_based"]]
TileSize = Union[int, Literal["SNR_based"]]


@dataclass
class NoiseModelConfig:
    """Sensor noise model. Values are read from DNG metadata when omitted."""

    alpha: Optional[float] = None
    beta: Optional[float] = None
    std_curve: tyro.conf.Suppress[List[float]] = field(default_factory=list, init=False, repr=False)
    diff_curve: tyro.conf.Suppress[List[float]] = field(default_factory=list, init=False, repr=False)


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
    do_color_correction: bool = True
    do_gamma_correction: bool = True
    do_tonemapping: bool = False
    sharpening: SharpeningConfig = field(default_factory=SharpeningConfig)
    do_devignetting: bool = False


@dataclass
class ExifConfig:
    cfa_pattern: List[List[int]]
    iso: float
    white_balance: List[float]


@dataclass
class Config:
    """Configuration for the handheld multi-frame super-resolution pipeline."""

    scale: float = 1
    mode: Literal["bayer", "grey"] = "bayer"
    debug: bool = False
    verbose: int = 1
    grey_method: Literal["FFT"] = "FFT"
    noise_model: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    exif: tyro.conf.Suppress[Optional[ExifConfig]] = field(default=None, init=False, repr=False)
