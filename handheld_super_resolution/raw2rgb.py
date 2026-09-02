from __future__ import annotations

import random
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import exifread
import numpy as np
from skimage import img_as_float32, filters

from .config import Config
from .utils_image import apply_orientation

if TYPE_CHECKING:
    from .utils_dng import DNGStack


SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# EXIF/DNG CalibrationIlluminant values that have a well-defined CCT.  LibRaw
# uses the daylight-side DNG matrix for its simple camera-to-sRGB transform.
# Picking the illuminant nearest D65 reproduces that choice while avoiding the
# very common ColorMatrix1-is-always-daylight assumption.
_ILLUMINANT_CCT = {
    1: 5500,   # Daylight
    3: 2850,   # Tungsten
    4: 5500,   # Flash (nominal)
    9: 5500,   # Fine weather
    10: 6500,  # Cloudy
    11: 7500,  # Shade
    17: 2856,  # Standard light A
    18: 4874,  # Standard light B
    19: 6774,  # Standard light C
    20: 5503,  # D55
    21: 6504,  # D65
    22: 7504,  # D75
    23: 5003,  # D50
    24: 3200,  # ISO studio tungsten
}


def _tag_scalar(tag: Any) -> int:
    value = tag.values[0] if hasattr(tag, "values") else tag
    return int(value)


def _tag_matrix(tag: Any) -> np.ndarray:
    values = [float(value.decimal()) for value in tag.values]
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 9:
        raise ValueError(f"Expected a 3x3 DNG color matrix, got {matrix.size} values")
    return matrix.reshape(3, 3)


def get_xyz2cam_from_exif(impath: str | Path) -> np.ndarray:
    """Read the daylight-side DNG XYZ-to-camera matrix.

    DNG does not guarantee that ``ColorMatrix1`` is the daylight matrix.  This
    function considers both matrix/illuminant pairs and chooses the reference
    illuminant nearest D65, matching LibRaw's matrix choice for the supported
    DNG path.
    """

    with open(impath, "rb") as raw_file:
        tags = exifread.process_file(raw_file, details=False)

    candidates = []
    for matrix_tag, illuminant_tag in (
        ("Image Tag 0xC621", "Image Tag 0xC65A"),
        ("Image Tag 0xC622", "Image Tag 0xC65B"),
    ):
        if matrix_tag not in tags:
            continue
        illuminant = _tag_scalar(tags[illuminant_tag]) if illuminant_tag in tags else None
        cct = _ILLUMINANT_CCT.get(illuminant)
        distance_from_d65 = abs(cct - 6504) if cct is not None else float("inf")
        candidates.append((distance_from_d65, _tag_matrix(tags[matrix_tag])))

    if not candidates:
        raise ValueError(f"No DNG ColorMatrix1/2 found in {impath}")

    return min(candidates, key=lambda candidate: candidate[0])[1].astype(np.float32)


def get_color_matrix(raw, xyz2cam=None):
    """Return the normalized linear-sRGB-to-camera matrix.

    This function keeps its historical direction for callers in the
    unprocessing pipeline. New rendering code should use
    :func:`get_camera_to_srgb_matrix`, whose direction is explicit.
    """

    if xyz2cam is None:
        xyz2cam = raw.rgb_xyz_matrix[:3]
    if np.linalg.norm(xyz2cam) == 0:
        warnings.warn("No camera color matrix found; using identity color conversion")
        return np.eye(3, dtype=np.float32)

    rgb2cam = np.asarray(xyz2cam, dtype=np.float64) @ SRGB_TO_XYZ
    row_sums = rgb2cam.sum(axis=-1, keepdims=True)
    if np.any(np.isclose(row_sums, 0.0)):
        raise ValueError("Cannot normalize a camera color matrix with a zero row sum")
    return (rgb2cam / row_sums).astype(np.float32)


def get_camera_to_srgb_matrix(raw=None, xyz2cam=None) -> np.ndarray:
    """Return a validated camera-RGB-to-linear-sRGB matrix.

    The independently derived DNG transform is the source of truth. When a
    RawPy object is supplied, LibRaw's processed ``color_matrix`` is used only
    after checking that it agrees with that transform. RawPy exposes LibRaw's
    vaguely named ``cmatrix`` rather than a documented camera-to-sRGB API.
    """

    if xyz2cam is None:
        if raw is None:
            raise ValueError("Either raw or xyz2cam must be provided")
        xyz2cam = np.asarray(raw.rgb_xyz_matrix[:3], dtype=np.float64)

    srgb_to_camera = get_color_matrix(raw, xyz2cam)
    metadata_matrix = np.linalg.inv(srgb_to_camera).astype(np.float32)

    if raw is None:
        return metadata_matrix

    libraw_matrix = np.asarray(raw.color_matrix, dtype=np.float32)[:3, :3]
    libraw_is_valid = (
        libraw_matrix.shape == (3, 3)
        and np.isfinite(libraw_matrix).all()
        and not np.allclose(libraw_matrix, 0.0)
        and abs(float(np.linalg.det(libraw_matrix))) > 1e-8
    )
    if libraw_is_valid and np.allclose(
        libraw_matrix, metadata_matrix, rtol=5e-4, atol=5e-4
    ):
        return libraw_matrix.copy()

    if libraw_is_valid:
        warnings.warn(
            "LibRaw's color_matrix disagrees with the DNG-derived camera-to-sRGB "
            "matrix; using the DNG-derived transform"
        )
    return metadata_matrix


def apply_ccm(image, ccm):
    """Apply a column-vector 3x3 color matrix to an HxWx3 image."""

    image = np.asarray(image)
    ccm = np.asarray(ccm)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {image.shape}")
    if ccm.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 color matrix, got shape {ccm.shape}")
    return np.einsum("ij,hwj->hwi", ccm, image)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Encode clipped linear sRGB values with the IEC sRGB transfer curve."""

    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )

def raw_to_rgb(raw, xyz2cam=None):
    return img_as_float32(raw.postprocess(use_camera_wb=True))


def postprocess(cam_rgb: np.ndarray, dng_stack: DNGStack, config: Config):
    """Render normalized, demosaiced camera RGB to display-ready sRGB.

    ``cam_rgb`` must already be black-subtracted and white-level-normalized;
    those sensor-domain operations intentionally remain in
    :meth:`DNGStack.get_raw_arrays`. The processing order here is:

    camera RGB -> white balance -> linear sRGB matrix -> optional sharpening
    -> optional sRGB transfer function.
    """

    rgb = np.asarray(cam_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected normalized camera RGB with shape HxWx3, got {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise ValueError("Camera RGB contains NaN or infinite values")

    postprocessing = config.postprocessing

    if postprocessing.do_white_balance:
        # RawPy exposes R/G1/B/G2. The SR output has one green channel, so use
        # G1 as the neutral reference and ignore the duplicate green gain.
        white_balance = np.asarray(dng_stack.white_balance[:3], dtype=np.float32)
        if white_balance.shape != (3,) or not np.isfinite(white_balance).all():
            raise ValueError(f"Invalid camera white balance: {white_balance!r}")
        if white_balance[1] <= 0:
            raise ValueError(f"Green white-balance gain must be positive: {white_balance!r}")
        white_balance = white_balance / white_balance[1]
        rgb = rgb * white_balance.reshape(1, 1, 3)

    if postprocessing.do_camera_to_linear_srgb:
        rgb = apply_ccm(rgb, dng_stack.camera_to_srgb)

    # Matrix conversion can produce valid negative/out-of-gamut values. This
    # simple display renderer clips them, just like the controlled RawPy
    # reference used in the tests.
    rgb = np.clip(rgb, 0.0, 1.0)

    sharpening = postprocessing.sharpening
    if sharpening is not None and sharpening.enabled:
        rgb = filters.unsharp_mask(
            rgb,
            radius=sharpening.radius,
            amount=sharpening.amount,
            channel_axis=2,
            preserve_range=True,
        )

    rgb = np.clip(rgb, 0.0, 1.0)
    if postprocessing.do_srgb_encoding:
        rgb = linear_to_srgb(rgb)

    # Applying image orientation
    if config.postprocessing.orientate_image:
        if 'Image Orientation' in dng_stack.tags.keys():
            ori = dng_stack.tags['Image Orientation'].values[0]
        else:
            ori = 1
            warnings.warn('The Image Orientation EXIF tag could not be found. \
                        The image may be mirrored or misoriented.')

        rgb = apply_orientation(rgb, ori)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
