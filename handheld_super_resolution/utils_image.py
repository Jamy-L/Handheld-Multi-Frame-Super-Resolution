import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve1d

from .config import GrayscaleMethod
from .utils import (
    DEFAULT_NUMPY_FLOAT_TYPE,
)


def apply_orientation(img, ori):
    """
    Applies an orientation to an image

    Parameters
    ----------
    img : numpy Array [ny, nx, c]
        Image
    ori : int
        Exif orientation as defined here:
            https://exiftool.org/TagNames/EXIF.html

    Returns
    -------
    Oriented image

    """

    if ori == 1:
        pass
    elif ori == 2:
        # Mirrored horizontal
        img = np.flip(img, axis=1)
    elif ori == 3:
        # Rotate 180
        img = np.rot90(img, k=2, axes=(0, 1))
    elif ori == 4:
        # Mirror vertical
        img = np.flip(img, axis=0)
    elif ori == 5:
        # Mirror horizontal and rotate 270 CW
        img = np.flip(img, axis=1)
        img = np.rot90(img, k=-3, axes=(0, 1))
    elif ori == 6:
        # Rotate 90 CW
        img = np.rot90(img, k=-1, axes=(0, 1))
    elif ori == 7:
        # Mirror horizontal and rotate 90 CW
        img = np.flip(img, axis=1)
        img = np.rot90(img, k=-1, axes=(0, 1))
    elif ori == 8:
        # Rotate 270 CW
        img = np.rot90(img, k=-3, axes=(0, 1))

    return img


def compute_grey_images(image: NDArray, method: GrayscaleMethod) -> NDArray[np.float32]:
    """Convert normalized RGGB Bayer data to NumPy gray for CPU alignment."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or any(size % 2 for size in image.shape):
        raise ValueError("Bayer input must be a 2D image with even dimensions")
    if method is GrayscaleMethod.DECIMATING:
        gray = (
            image[::2, ::2] + image[::2, 1::2] + image[1::2, ::2] + image[1::2, 1::2]
        ) * 0.25
        weights = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        gray = convolve1d(
            convolve1d(gray, weights, axis=0, mode="nearest"),
            weights,
            axis=1,
            mode="nearest",
        )
    elif method is GrayscaleMethod.DEMOSAICING:
        gray = demosaic_to_grey(image)
    else:
        raise ValueError(f"Unknown grayscale method: {method}")
    # Demosaicing can overshoot the normalized sensor range; the alignment API
    # requires finite grayscale samples in [0, 1].
    return np.ascontiguousarray(np.clip(gray, 0, 1), dtype=np.float32)


def demosaic_to_grey(img: NDArray) -> NDArray[np.float32]:
    """Demosaic a normalized RGGB CFA and collapse its RGB channels to gray."""
    if img.ndim != 2:
        raise ValueError(f"Bayer input must be 2D, got shape {img.shape}")
    try:
        from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    except ImportError as error:
        raise ImportError(
            "The demosaicing alignment method requires colour-demosaicing"
        ) from error

    rgb = demosaicing_CFA_Bayer_Menon2007(img, pattern="RGGB")
    gray = 0.25 * (
        rgb[:, :, 0] + 2 * rgb[:, :, 1] + rgb[:, :, 2]
    )  # Luma conversion, consistent with quad averaging
    return np.ascontiguousarray(gray, dtype=DEFAULT_NUMPY_FLOAT_TYPE)


def cfa_to_rggb(x: np.ndarray, source_cfa: np.ndarray):
    assert x.ndim in (2, 3), f"Expected 2 or 3 dim, got {x.shape}"
    assert source_cfa.shape == (2, 2), (
        f"expected cfa of shape (2, 2), got {source_cfa.shape}"
    )

    if np.array_equal(source_cfa, np.array([[0, 1], [1, 2]])):
        return x

    # BGGR
    if np.array_equal(source_cfa, np.array([[2, 1], [1, 0]])):
        return np.flip(x, axis=(-1, -2))

    # GBRG
    if np.array_equal(source_cfa, np.array([[1, 0], [2, 1]])):
        return np.flip(x, axis=-1)
    # GRGB
    if np.array_equal(source_cfa, np.array([[1, 2], [0, 1]])):
        return np.flip(x, axis=-2)

    raise NotImplementedError(f"Unsupported CFA pattern {source_cfa}")


def rggb_to_cfa(x: np.ndarray, target_cfa: np.ndarray):
    # the function is its own inverse...
    return cfa_to_rggb(x, target_cfa)


def estimate_image_snr(
    ref_img: NDArray[np.float32],
    alpha: tuple[float, float, float, float],
    beta: tuple[float, float, float, float],
) -> float:
    r = ref_img[::2, ::2]
    g1 = ref_img[::2, 1::2]
    g2 = ref_img[1::2, ::2]
    b = ref_img[1::2, 1::2]

    varr = alpha[0] * r + beta[0]
    varg1 = alpha[1] * g1 + beta[1]
    varg2 = alpha[3] * g2 + beta[3]
    varb = alpha[2] * b + beta[2]

    validr = (r > 0) & (r < 1)
    validg1 = (g1 > 0) & (g1 < 1)
    validg2 = (g2 > 0) & (g2 < 1)
    validb = (b > 0) & (b < 1)

    top = np.concatenate((r[validr], g1[validg1], g2[validg2], b[validb]))
    bot = np.concatenate((varr[validr], varg1[validg1], varg2[validg2], varb[validb]))

    return float(np.sqrt(np.sum(top * top) / np.sum(bot)))
