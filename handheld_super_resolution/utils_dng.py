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

from . import raw2rgb
from .utils import DEFAULT_NUMPY_FLOAT_TYPE

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

def load_dng_burst(burst_path):
    """
    Loads a dng burst into numpy arrays, and their exif tags.

    Parameters
    ----------
    burst_path : Path or str
        Path of the folder containing the .dngs

    Returns
    -------
    ref_raw : numpy Array[H, W]
        Reference frame
    raw_comp : numpy Array[n, H, W]
        Stack of non-reference frame
    ISO : int
        Clipped ISO (between 100 and 3600)
    tags : dict
        Tags of the reference frame
    CFA : numpy array [2, 2]
        Bayer pattern of the stack
    xyz2cam : Array
        The xyz to camera color matrix
    reference_path
        Path of the reference image.

    """
    ref_id = 0
    raw_comp = []

    # This ensures that burst_path is a Path object
    burst_path = Path(burst_path)


    #### Read dng as numpy arrays
    # Get the list of raw images in the burst path
    raw_path_list = glob.glob(os.path.join(burst_path.as_posix(), '*.dng'))
    assert len(raw_path_list) != 0, 'At least one raw .dng file must be present in the burst folder.'
    # Read the raw bayer data from the DNG files
    for index, raw_path in enumerate(raw_path_list):
        with rawpy.imread(raw_path) as rawObject:
            if index != ref_id:

                raw_comp.append(rawObject.raw_image.copy())  # copy otherwise image data is lost when the rawpy object is closed
    raw_comp = np.array(raw_comp)

    # Reference image selection and metadata
    raw = rawpy.imread(raw_path_list[ref_id])
    ref_raw = raw.raw_image.copy()




    #### Reading tags of the reference image
    xyz2cam = raw2rgb.get_xyz2cam_from_exif(raw_path_list[ref_id])

    # reading exifs for white level, black leve and CFA
    with open(raw_path_list[ref_id], 'rb') as raw_file:
        tags = exifread.process_file(raw_file)


    if 'Image PhotometricInterpretation' in tags.keys():
        photo_inter = tags['Image PhotometricInterpretation'].values[0]
        if photo_inter not in SUPPORTED:
            warnings.warn('The input images have a photometric interpretation '\
                             'of type "{}", but only {} are supprted.'.format(
                                 PHOTO_INTER[photo_inter], str([PHOTO_INTER[i] for i in SUPPORTED])))
            
    else:
        warnings.warn('PhotometricInterpretation could not be found in image tags. '\
                     'Please ensure that it is one of {}'.format(str([PHOTO_INTER[i] for i in SUPPORTED])))
            

    white_level = int(raw.white_level)  # there is only one white level
    # exifread method is inconsistent because camera manufacters can put
    # this under many different tags.

    black_levels = raw.black_level_per_channel

    white_balance = raw.camera_whitebalance

    CFA = raw.raw_pattern.copy() # copying to ensure contiguity of the array
    CFA[CFA == 3] = 1 # Rawpy gives channel 3 to the second green channel. Setting both greens to 1

    if 'EXIF ISOSpeedRatings' in tags.keys():
        ISO = int(str(tags['EXIF ISOSpeedRatings']))
    elif 'Image ISOSpeedRatings' in tags.keys():
        ISO = int(str(tags['Image ISOSpeedRatings']))
    else:
        raise AttributeError('ISO value could not be found in both EXIF and Image type.')

    # Clipping ISO to 100 from below
    ISO = max(100, ISO)
    ISO = min(3200, ISO)




    #### Performing whitebalance and normalizing into 0, 1

    if np.issubdtype(type(ref_raw[0, 0]), np.integer):
        # Here do black and white level correction and white balance processing for all image in comp_images
        # Each image in comp_images should be between 0 and 1.
        # ref_raw is a (H,W) array
        ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        for i in range(2):
            for j in range(2):
                channel = CFA[i, j]
                ref_raw[i::2, j::2] = (ref_raw[i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
                ref_raw[i::2, j::2] *= white_balance[channel] / white_balance[1]

        ref_raw = np.clip(ref_raw, 0.0, 1.0)
        # The division by the green WB value is important because WB may come with integer coefficients instead

    if np.issubdtype(type(raw_comp[0, 0, 0]), np.integer):
        raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
        # raw_comp is a (N, H,W) array
        for i in range(2):
            for j in range(2):
                channel = channel = CFA[i, j]
                raw_comp[:, i::2, j::2] = (raw_comp[:, i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
                raw_comp[:, i::2, j::2] *= white_balance[channel] / white_balance[1]
        raw_comp = np.clip(raw_comp, 0., 1.)

    return ref_raw, raw_comp, ISO, tags, CFA, xyz2cam, raw_path_list[ref_id]


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
