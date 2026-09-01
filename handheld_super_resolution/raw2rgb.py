from __future__ import annotations

import random
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import exifread
import numpy as np
from skimage import img_as_float32, filters

import cv2

from .config import Config

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



def get_random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
               [-0.5625, 1.6328, -0.0469],
               [-0.0703, 0.2188, 0.6406]],
              [[0.4913, -0.0541, -0.0202],
               [-0.613, 1.3513, 0.2906],
               [-0.1564, 0.2151, 0.7183]],
              [[0.838, -0.263, -0.0639],
               [-0.2887, 1.0725, 0.2496],
               [-0.0627, 0.1427, 0.5438]],
              [[0.6596, -0.2079, -0.0562],
               [-0.4782, 1.3016, 0.1933],
               [-0.097, 0.1581, 0.5181]]]

    num_ccms = len(xyz2cams)  # (4,3,3)

    weights = np.random.rand(num_ccms).reshape((num_ccms, 1, 1))
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(axis=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2cam = xyz2cam @ SRGB_TO_XYZ

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(axis=-1, keepdim=True)
    return rgb2cam


def get_random_noise_parameters(log_min_shot=0.0001, log_max_shot=0.012, sigma_read_noise=0.26):
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(log_min_shot)
    log_max_shot_noise = math.log(log_max_shot)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=sigma_read_noise)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def get_random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / random.gauss(mu=0.8, sigma=0.1)

    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.ndim == 3 and image.shape[2] == 3

    gains = np.array([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains.reshape((1, 1, 3))

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = np.mean(image, axis=-1, keepdims=True)
    inflection = 0.9
    mask = ((gray - inflection).cllp(min=0.0) / (1.0 - inflection))
    mask = mask * mask

    safe_gains = np.max(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains


def get_color_matrix(raw, xyz2cam=None):
    """Return the normalized linear-sRGB-to-camera matrix.

    This function keeps its historical direction for callers in the
    unprocessing pipeline. New rendering code should use
    :func:`get_camera_to_srgb_matrix`, whose direction is explicit.
    """

    if xyz2cam is None:
        xyz2cam = raw.rgb_xyz_matrix[:3]
    if np.linalg.norm(xyz2cam) == 0:
        warnings.warn("No camera color matrix found; using identity color correction")
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


def gamma_compression(img, gamma=2.2):
    img = np.clip(img, a_min=0.0, a_max=1.0)
    return img**(1./gamma)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Encode clipped linear sRGB values with the IEC sRGB transfer curve."""

    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def gamma_expansion(img, gamma=2.2):
    img = np.clip(img, a_min=1e-8, a_max=1.0)
    return img ** gamma


def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    # image_out = 3 * image**2 - 2 * image**3
    
    # tonemap = cv2.createTonemap(1.0)
    # image_out = tonemap.process(image)
    
    from skimage import img_as_ubyte, img_as_float32
    times = [1, 0.5, 2]
    images = [img_as_ubyte(np.clip(image*i, 0, 1)) for i in times] 
    
    
    merge_mertens = cv2.createMergeMertens()
    image_out = merge_mertens.process(images)
    image_out = img_as_float32(image_out)

    image_out = 3 * image_out**2 - 2 * image_out**3
    return image_out


def invert_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = np.clip(image, a_min=0.0, a_max=1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)


def unprocess_isp(jpg, log_max_shot=0.012):
    """
    Convert a jpg image to raw image.
    """
    rgb2cam = get_random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = get_random_gains()
    lambda_read, lambda_shot = get_random_noise_parameters(log_max_shot=log_max_shot)
    metadata = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                'blue_gain': blue_gain, 'lambda_shot': lambda_shot, 'lambda_read': lambda_read}

    ## Inverse tone mapping
    jpg = invert_smoothstep(jpg)

    ## Gamma expansion
    jpg = gamma_expansion(jpg)

    ## Inverse color matrix
    raw = apply_ccm(jpg, rgb2cam)

    ## Inverse gains
    raw = safe_invert_gains(raw, red_gain, blue_gain, rgb_gain)

    return raw, metadata


def raw_to_rgb(raw, xyz2cam=None):
    return img_as_float32(raw.postprocess(use_camera_wb=True))


def postprocess(cam_rgb: np.ndarray, dng_stack: DNGStack, config: Config):
    """Render normalized, demosaiced camera RGB to display-ready sRGB.

    ``cam_rgb`` must already be black-subtracted and white-level-normalized;
    those sensor-domain operations intentionally remain in
    :meth:`DNGStack.get_raw_arrays`. The processing order here is:

    camera RGB -> white balance -> linear sRGB matrix -> optional tone/detail
    operations -> optional sRGB transfer function.
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

    if postprocessing.do_color_correction:
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

    if postprocessing.do_tonemapping:
        rgb = apply_smoothstep(rgb)

    if postprocessing.do_devignetting:
        raise NotImplementedError

    rgb = np.clip(rgb, 0.0, 1.0)
    if postprocessing.do_gamma_correction:
        rgb = linear_to_srgb(rgb)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
