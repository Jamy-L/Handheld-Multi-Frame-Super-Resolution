"""Command-line interface for the handheld super-resolution pipeline."""

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import tyro
from skimage import img_as_ubyte

from .config import Config


@dataclass
class CliConfig:
    """Run handheld multi-frame super-resolution on a DNG burst."""

    impath: Path
    """Directory containing the input burst."""

    outpath: Path
    """Output image path (.png or .dng)."""

    config: Annotated[Config, tyro.conf.arg(name="")] = field(default_factory=Config)


def parse_args(args=None):
    return tyro.cli(CliConfig, args=args)


def _save_image(path, rgb_8bit_data):
    return cv2.imwrite(str(path), cv2.cvtColor(rgb_8bit_data, cv2.COLOR_RGB2BGR))


def main(args=None):
    options = parse_args(args)
    config = options.config

    if options.outpath.suffix == ".dng":
        config.postprocessing.enabled = False

    # Delay GPU-heavy imports so parsing and --help remain lightweight.
    from .super_resolution import process

    output, debug = process(options.impath, config)
    output = np.clip(np.nan_to_num(output), 0, 1)

    if options.outpath.suffix == ".dng":
        from .utils_dng import save_as_dng

        if config.verbose >= 1:
            print(f"Saving output to {options.outpath}")
        reference = glob.glob(os.path.join(options.impath, "*.dng"))[0]
        save_as_dng(output, reference, options.outpath)
    else:
        _save_image(options.outpath, img_as_ubyte(output))

    accumulated = debug.get("accumulated robustness")
    if config.robustness.save_mask and accumulated is not None:
        image_count = len(glob.glob(os.path.join(options.impath, "*.dng")))
        robustness = accumulated / (image_count - 1)
        robustness = np.repeat(robustness[..., None], 3, axis=-1)
        robustness = cv2.resize(
            robustness,
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        _save_image(options.outpath.with_suffix(".rob.png"), img_as_ubyte(robustness))
