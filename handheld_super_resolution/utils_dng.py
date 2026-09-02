# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:44:36 2023

@author: jamyl
"""

import os
import glob
import subprocess

import numpy as np
from pathlib import Path
import exifread
import rawpy
import imageio
import warnings

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, List

from . import raw2rgb
from .utils import DEFAULT_NUMPY_FLOAT_TYPE
from .utils_image import cfa_to_rggb

# Paths of exiftool and dng validate. Only necessary to output dng.
EXIFTOOL_PATH = 'exiftool' # Assumes exiftool is in PATH, but you can also paste the path here
DNG_VALIDATE_PATH = 'dng_validate' # Same applies here


# See "PhotometricInterpretation in" https://exiftool.org/TagNames/EXIF.html
PHOTO_INTER = {
    0 : 'WhiteIsZero',
    1 : 'BlackIsZero',
    2 : 'RGB',
    3 : 'RGB Palette',
    4 : 'Transparency Mask',
    5 : 'CMYK',
    6 : 'YCbCr',
    8 : 'CIELab',
    9 : 'ICCLab',
    10 : 'ITULab',
    32803 : 'Color Filter Array',
    32844 : 'Pixar LogL',
    32845 : 'Pixar LogLuv',
    32892 : 'Sequential Color Filter',
    34892 : 'Linear Raw',
    51177 : 'Depth Map',
    52527 : 'Semantic Mask'}

# Supported Photometric Interpretations
SUPPORTED = [1, 32803]


def expand_noise_profile_to_rgbg(alpha, beta) -> Tuple[Tuple[float, float, float, float],
                                                        Tuple[float, float, float, float]]:
    """Expand one- or three-plane profiles to R, G1, B, G2 order."""
    alpha = tuple(float(value) for value in alpha)
    beta = tuple(float(value) for value in beta)
    if len(alpha) != len(beta):
        raise ValueError(
            f"alpha and beta must contain the same number of values, got "
            f"{len(alpha)} and {len(beta)}"
        )
    if len(alpha) == 1:
        return alpha * 4, beta * 4
    if len(alpha) == 3:
        return (alpha[0], alpha[1], alpha[2], alpha[1]), (
            beta[0], beta[1], beta[2], beta[1]
        )
    if len(alpha) == 4:
        return alpha, beta
    raise ValueError(
        f"noise profiles must contain 1, 3, or 4 values, got {len(alpha)}"
    )


def noise_profile_from_tags(tags) -> Tuple[Tuple[float, float, float, float],
                                            Tuple[float, float, float, float]]:
    """Read the DNG NoiseProfile tag in R, G1, B, G2 color-plane order."""
    tag = tags.get('Image Tag 0xC761')
    if tag is None:
        raise ValueError("DNG does not contain a NoiseProfile (0xC761) tag")
    values = [float(value[0]) for value in tag.values]
    if len(values) % 2:
        raise ValueError(f"DNG NoiseProfile must contain alpha/beta pairs, got {values!r}")
    return expand_noise_profile_to_rgbg(values[::2], values[1::2])


def read_dng_noise_profile(path: Union[str, Path]) -> Tuple[
    Tuple[float, float, float, float], Tuple[float, float, float, float]
]:
    """Read only the noise profile needed by the Monte Carlo CLI."""
    path = Path(path)
    with path.open('rb') as raw_file:
        tags = exifread.process_file(raw_file, details=False)
    return noise_profile_from_tags(tags)


@dataclass
class DNGStack:
    """Images, metadata, calibration values, and paths for a DNG burst."""

    burst_path: Path
    raw_paths: Tuple[Path, ...]
    reference_index: int
    ref_raw: np.ndarray
    raw_comp: np.ndarray
    iso: int
    tags: Dict[str, Any]
    cfa: np.ndarray
    xyz2cam: np.ndarray
    camera_to_srgb: np.ndarray
    white_balance: np.ndarray
    white_level: int
    black_levels: np.ndarray
    alpha: Tuple[float, float, float, float]
    beta: Tuple[float, float, float, float]
    photometric_interpretation: Union[int, None] = None

    @property
    def reference_path(self) -> Path:
        """Path of the frame used as the reference image."""
        return self.raw_paths[self.reference_index]

    @property
    def comparison_paths(self) -> Tuple[Path, ...]:
        """Paths of all non-reference frames, in stack order."""
        return self.raw_paths[:self.reference_index] + self.raw_paths[self.reference_index + 1:]

    def get_raw_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        # Array RGGB by design
        # wb is given as r g1 b g2
        ref = self.ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        ref[::2, ::2] = (ref[::2, ::2] - self.black_levels[0]) / (self.white_level - self.black_levels[0])
        ref[::2, 1::2] = (ref[::2, 1::2] - self.black_levels[1]) / (self.white_level - self.black_levels[1])
        ref[1::2, ::2] = (ref[1::2, ::2] - self.black_levels[3]) / (self.white_level - self.black_levels[3])
        ref[1::2, 1::2] = (ref[1::2, 1::2] - self.black_levels[2]) / (self.white_level - self.black_levels[2])

        comp = self.raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        comp[:, ::2, ::2] = (comp[:, ::2, ::2] - self.black_levels[0]) / (self.white_level - self.black_levels[0])
        comp[:, ::2, 1::2] = (comp[:, ::2, 1::2] - self.black_levels[1]) / (self.white_level - self.black_levels[1])
        comp[:, 1::2, ::2] = (comp[:, 1::2, ::2] - self.black_levels[3]) / (self.white_level - self.black_levels[3])
        comp[:, 1::2, 1::2] = (comp[:, 1::2, 1::2] - self.black_levels[2]) / (self.white_level - self.black_levels[2])
        return ref, comp


def load_dng_burst(burst_path: Union[str, Path]) -> DNGStack:
    """
    Loads a dng burst into numpy arrays, and their exif tags.

    Parameters
    ----------
    burst_path : Path or str
        Path of the folder containing the .dngs

    Returns
    -------
    DNGStack
        Loaded frames, metadata, calibration values, and source paths.

    """
    ref_id = 0
    raw_comp = []

    burst_path = Path(burst_path)


    #### Read dng as numpy arrays
    raw_path_list = sorted(glob.glob(os.path.join(burst_path.as_posix(), '*.dng')))
    assert len(raw_path_list) != 0, 'At least one raw .dng file must be present in the burst folder.'

    for index, raw_path in enumerate(raw_path_list):
        with rawpy.imread(raw_path) as rawObject:
            if index != ref_id:
                raw_comp.append(rawObject.raw_image.copy())  # copy otherwise image data is lost when the rawpy object is closed
    raw_comp = np.array(raw_comp)

    raw = rawpy.imread(raw_path_list[ref_id])
    ref_raw = raw.raw_image.copy()


    #### Reading tags of the reference image
    xyz2cam = raw2rgb.get_xyz2cam_from_exif(raw_path_list[ref_id])
    camera_to_srgb = raw2rgb.get_camera_to_srgb_matrix(raw, xyz2cam)

    # reading exifs for white level, black level and CFA
    with open(raw_path_list[ref_id], 'rb') as raw_file:
        tags = exifread.process_file(raw_file)

    photometric_interpretation = tags.get('Image PhotometricInterpretation', None)
    if photometric_interpretation:
        photometric_interpretation = photometric_interpretation.values[0]
        if photometric_interpretation not in SUPPORTED:
            warnings.warn('The input images have a photometric interpretation '\
                             'of type "{}", but only {} are supprted.'.format(
                                 PHOTO_INTER[photometric_interpretation], str([PHOTO_INTER[i] for i in SUPPORTED])))
            
    else:
        warnings.warn('PhotometricInterpretation could not be found in image tags. '\
                     'Please ensure that it is one of {}'.format(str([PHOTO_INTER[i] for i in SUPPORTED])))
            

    white_level = int(raw.white_level)  # there is only one white level
    # exifread method is inconsistent because camera manufacters can put
    # this under many different tags.

    if raw.color_desc != b"RGBG": # This has nothing to do with the CFA, just the order of the color wb channels
        raise NotImplementedError(f"Unexpected color_desc: {raw.color_desc!r}")
    black_levels = np.asarray(raw.black_level_per_channel) # R, G1, B, G2
    white_balance = np.asarray(raw.camera_whitebalance) # R G1, B, G2
    assert len(white_balance) == 4, f"Unexpected camera_whitebalance: {white_balance!r}"
    if white_balance[-1] == 0:
        warnings.warn(f"Camera white balance ends with 0: {white_balance!r}. Using the second green channel's value instead.")
        white_balance[-1] = white_balance[1]

    CFA = raw.raw_pattern.copy() # copying to ensure contiguity of the array
    CFA[CFA == 3] = 1 # Rawpy gives channel 3 to the second green channel. Setting both greens to 1
    raw.close()

    if 'EXIF ISOSpeedRatings' in tags.keys():
        iso = int(str(tags['EXIF ISOSpeedRatings']))
    elif 'Image ISOSpeedRatings' in tags.keys():
        iso = int(str(tags['Image ISOSpeedRatings']))
    else:
        raise AttributeError('ISO value could not be found in both EXIF and Image type.')

    # Clipping ISO to 100 from below
    iso = max(100, iso)
    iso = min(3200, iso)

    alpha, beta = noise_profile_from_tags(tags)


    #### Performing whitebalance
    assert (type_ := type(ref_raw[0, 0])) == (y := type(raw_comp[0, 0, 0])), f'Reference and comp images should have the same data type, got {type_} and {y}.'
    assert np.issubdtype(type_, np.integer), f'Input DNG images are not in integer format: is the input valid RAW data? Got {type_}.'

    # if np.issubdtype(type_, np.integer):
    #     ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
    #     raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
    #     for i in range(2):
    #         for j in range(2):
    #             channel = CFA[i, j]
    #             k = white_balance[channel] / white_balance[1]
    #             ref_raw[i::2, j::2] = (ref_raw[i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
    #             raw_comp[:, i::2, j::2] = (raw_comp[:, i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
    #             ref_raw[i::2, j::2] *= k
    #             raw_comp[:, i::2, j::2] *= k
    # else:
    #     warnings.warn('Input DNG images are not in integer format: is the input valid RAW data?')

    # Flip to rggb
    ref_raw = cfa_to_rggb(ref_raw, CFA)
    raw_comp = cfa_to_rggb(raw_comp, CFA)

    return DNGStack(
        burst_path=burst_path,
        raw_paths=tuple(Path(raw_path) for raw_path in raw_path_list),
        reference_index=ref_id,
        ref_raw=ref_raw,
        raw_comp=raw_comp,
        iso=iso,
        tags=tags,
        cfa=CFA,
        xyz2cam=xyz2cam,
        camera_to_srgb=camera_to_srgb,
        white_balance=white_balance,
        white_level=white_level,
        black_levels=black_levels,
        alpha=alpha,
        beta=beta,
        photometric_interpretation=photometric_interpretation,
    )


def save_as_dng(np_img, ref_dng_path, outpath):
    '''
    Saves a RGB numpy image as dng.
    The image is first saved as 16bits tiff, then the extension is swapped
    to .dng. The metadata are then overwritten using a reference dng, and
    the final dng is built using dng_validate.

    Requires :
    - dng_validate (can be found in dng sdk):
        https://helpx.adobe.com/camera-raw/digital-negative.html#dng_sdk_download

    - exiftool
        https://exiftool.org/


    Based on :
    https://github.com/gluijk/dng-from-tiff/blob/main/dngmaker.bat
    https://github.com/antonwolf/dng_stacker/blob/master/dng_stacker.bat

    Parameters
    ----------
    np_img : numpy array
        RGB image
    rawpy_ref : _rawpy.Rawpy 
        image containing some relevant tags
    outpath : Path
        output save path.

    Returns
    -------
    None.

    '''
    assert np_img.ndim == 3 and np_img.shape[-1] == 3, f"Got {np_img.shape}, expected HxWx3 RGB image."

    np_int_img = np.copy(np_img)  # copying to avoid inplace-overwritting

    raw = rawpy.imread(ref_dng_path)
    white_balance = raw.camera_whitebalance
    white_balance = [x/white_balance[1] for x in white_balance]  # Normalize to green channel

    # Quantize to 16 bits using full range
    new_white_level = 2**16 - 1
    new_black_level = 0

    np_int_img = np_int_img * (new_white_level - new_black_level) + new_black_level
    np_int_img = np.round(np_int_img)

    np_int_img = np.clip(np_int_img, 0, new_white_level).astype(np.uint16)

    #### Saving the image as 16 bits RGB tiff
    save_as_tiff(np_int_img, outpath)

    tmp_path = outpath.parent / 'tmp.dng'

    # Deleting tmp.dng if it is already existing
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    #### Overwritting the tiff tags with dng tags, and replacing the .tif extension
    # by .dng
    cmd = [
        EXIFTOOL_PATH,
        "-n",
        "-IFD0:SubfileType#=0",
        # "-DNGBackwardVersion=1 2 0 0"
        "-IFD0:PhotometricInterpretation#=34892",
        "-BaselineExposure=0",
        "-SamplesPerPixel#=3",
        "-overwrite_original",
        "-tagsfromfile", ref_dng_path,
        "-all:all>all:all",
        "-DNGVersion",
        "-DNGBackwardVersion",
        "-ColorMatrix1",
        "-ColorMatrix2",
        "-IFD0:CalibrationIlluminant1<SubIFD:CalibrationIlluminant1",
        "-IFD0:CalibrationIlluminant2<SubIFD:CalibrationIlluminant2",
        f"-AsShotNeutral=1 1 1",
        # "-IFD0:BlackLevelRepeatDim<SubIFD:BlackLevelRepeatDim",
        # "-IFD0:CFARepeatPatternDim<SubIFD:CFARepeatPatternDim",
        # "-IFD0:CFAPattern2<SubIFD:CFAPattern2",
        # "-IFD0:ActiveArea<SubIFD:ActiveArea",
        # "-IFD0:DefaultScale<SubIFD:DefaultScale",
        # "-IFD0:DefaultCropOrigin<SubIFD:DefaultCropOrigin",
        # "-IFD0:DefaultCropSize<SubIFD:DefaultCropSize",
        "-IFD0:OpcodeList1<SubIFD:OpcodeList1",
        "-IFD0:OpcodeList2<SubIFD:OpcodeList2",
        "-IFD0:OpcodeList3<SubIFD:OpcodeList3",
        "-o", tmp_path.as_posix(),
        outpath.with_suffix('.tif').as_posix()
    ]

    # Run the command safely
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ExifTool command failed: {result.stderr}")
    else:
        print("ExifTool succeeded")
        print(result.stdout)

    # Adding further tags that cant be set during first run (because it was a .tiff and now it's a .dng)
    exiftool_args = [
        EXIFTOOL_PATH,
        "-n",
        "-overwrite_original",
        "-tagsfromfile", ref_dng_path,
        f"-IFD0:AnalogBalance={white_balance[0]} {white_balance[1]} {white_balance[2]}",
        f"-AnalogBalance={white_balance[0]} {white_balance[1]} {white_balance[2]}",
        "-AsShotWhiteXY=",
        "-BlackLevelDeltaH=",
        "-BlackLevelDeltaV=",
        "-XMP:ColorTemperature=",
        "-IFD0:ColorMatrix1",
        "-IFD0:ColorMatrix2",
        "-IFD0:CameraCalibration1",
        "-IFD0:CameraCalibration2",
        "-IFD0:ProfileHueSatMap1",
        "-IFD0:ProfileHueSatMap2",
        "-IFD0:ProfileLookTable"
        f"-IFD0:AsShotNeutral=1 1 1",
        f"-AsShotNeutral=1 1 1",
        f"-IFD0:WhiteLevel={new_white_level} {new_white_level} {new_white_level}",
        f"-IFD0:BlackLevel={new_black_level} {new_black_level} {new_black_level}",
        f"-BlackLevel={new_black_level} {new_black_level} {new_black_level}",
        f"-WhiteLevel={new_white_level} {new_white_level} {new_white_level}",
        "-IFD0:BaselineExposure",
        "-IFD0:CalibrationIlluminant1",
        "-IFD0:CalibrationIlluminant2",
        "-IFD0:ForwardMatrix1",
        "-IFD0:ForwardMatrix2",
        tmp_path.as_posix(),
    ]

    result = subprocess.run(exiftool_args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ExifTool failed:\n{result.stderr}")
    else:
        print(result.stdout)

    # Running DNG_validate
    cmd = [
        DNG_VALIDATE_PATH,
        "-16",
        "-dng",
        outpath.with_suffix(".dng").as_posix(),
        tmp_path.as_posix(),
    ]

    # Use Popen to stream output in real-time
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="")  # print each line as it arrives
        proc.wait()  # wait for completion
        if proc.returncode != 0:
            raise RuntimeError(f"DNG_validate failed with return code {proc.returncode}")

    os.remove(tmp_path)


def save_as_tiff(int_im, outpath):
    # 16 bits uncompressed by default
    # Imageio is the only module I could find to save 16 bits RGB tiffs without compression (cv2 does LZW).
    # It is vital to have uncompressed image, because validate_dng cannot work if the tiff is compressed.
    try:
        # Try to write as classic TIFF
        with imageio.imopen(outpath.with_suffix('.tif').as_posix(), 'w', bigtiff=False) as img_file: # Cant put bigtiff=True, else exiftool wont work to write tags...
            img_file.write(int_im)
    except ValueError as e:
        # ImageIO raises ValueError if data too large for classic TIFF (> 4GB)
        raise RuntimeError(
            f"Failed to write '{outpath.name}' as a classic TIFF. "
            f"The image is too large for bigtiff=False. "
            f"Raise an issue on github if you need support for bigtiff."
        ) from e
