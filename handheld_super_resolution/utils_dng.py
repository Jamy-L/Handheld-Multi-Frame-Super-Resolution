# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:44:36 2023

@author: jamyl
"""

import os
import glob

import numpy as np
from pathlib import Path
import exifread
import rawpy
import imageio

from . import raw2rgb
from .utils import DEFAULT_NUMPY_FLOAT_TYPE

# Paths of exiftool and dng validate. Only necessary to output dng.
EXIFTOOL_PATH = 'path/to/exiftool.exe'
DNG_VALIDATE_PATH = 'path/to/dng_validate.exe'


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
    xyz2cam : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    ref_id = 0  # TODO Select ref id based on HDR+ method
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
    assert np_img.shape[-1] == 3

    np_int_img = np.copy(np_img)  # copying to avoid inplace-overwritting
    #### Undo White balance and black level
    # get tags
    with open(ref_dng_path, 'rb') as raw_file:
        tags = exifread.process_file(raw_file)

    black_levels = tags['Image BlackLevel']
    if isinstance(black_levels.values[0], int):
        black_levels = np.array(black_levels.values)
    else:  # Sometimes this tag is a fraction object for some reason. It seems that black levels are all integers anyway
        black_levels = np.array([int(x.decimal()) for x in black_levels.values])

    raw = rawpy.imread(ref_dng_path)
    white_balance = raw.camera_whitebalance

    # Reverse WB
    new_white_level = 2**16 - 1

    for c in range(3):
        np_int_img[:, :, c] /= white_balance[c] / white_balance[1]
        np_int_img[:, :, c] = np_int_img[:, :, c] * (new_white_level - black_levels[c]) + black_levels[c]

    np_int_img = np.clip(np_int_img, 0, 2**16 - 1).astype(np.uint16)

    #### Saving the image as 16 bits RGB tiff
    save_as_tiff(np_int_img, outpath)

    tmp_path = outpath.parent / 'tmp.dng'

    # Deleting tmp.dng if it is already existing
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    #### Overwritting the tiff tags with dng tags, and replacing the .tif extension
    # by .dng
    cmd = '''
        {} -n\
        -IFD0:SubfileType#=0\
        -IFD0:PhotometricInterpretation#=34892\
        -SamplesPerPixel#=3\
        -overwrite_original -tagsfromfile {}\
        "-all:all>all:all"\
        -DNGVersion\
        -DNGBackwardVersion\
        -ColorMatrix1 -ColorMatrix2\
        "-IFD0:BlackLevelRepeatDim<SubIFD:BlackLevelRepeatDim"\
        "-IFD0:CalibrationIlluminant1<SubIFD:CalibrationIlluminant1"\
        "-IFD0:CalibrationIlluminant2<SubIFD:CalibrationIlluminant2"\
        "-IFD0:CFARepeatPatternDim<SubIFD:CFARepeatPatternDim"\
        "-IFD0:CFAPattern2<SubIFD:CFAPattern2"\
        -AsShotNeutral\
        "-IFD0:ActiveArea<SubIFD:ActiveArea"\
        "-IFD0:DefaultScale<SubIFD:DefaultScale"\
        "-IFD0:DefaultCropOrigin<SubIFD:DefaultCropOrigin"\
        "-IFD0:DefaultCropSize<SubIFD:DefaultCropSize"\
        "-IFD0:OpcodeList1<SubIFD:OpcodeList1"\
        "-IFD0:OpcodeList2<SubIFD:OpcodeList2"\
        "-IFD0:OpcodeList3<SubIFD:OpcodeList3"\
         -o {} {}
        '''.format(EXIFTOOL_PATH, ref_dng_path,
                   tmp_path.as_posix(),
                   outpath.with_suffix('.tif').as_posix())
    os.system(cmd)

    # adding further dng tags
    cmd = """
        {} -n -overwrite_original -tagsfromfile {}\
        "-IFD0:AnalogBalance"\
        "-IFD0:ColorMatrix1" "-IFD0:ColorMatrix2"\
        "-IFD0:CameraCalibration1" "-IFD0:CameraCalibration2"\
        "-IFD0:AsShotNeutral" "-IFD0:BaselineExposure"\
        "-IFD0:CalibrationIlluminant1" "-IFD0:CalibrationIlluminant2"\
        "-IFD0:ForwardMatrix1" "-IFD0:ForwardMatrix2"\
        {}\
        """.format(EXIFTOOL_PATH, ref_dng_path, tmp_path.as_posix())
    os.system(cmd)

    # Running DNG_validate
    cmd = """
    {} -16 -dng\
    {}\
    {}\
    """.format(DNG_VALIDATE_PATH, outpath.with_suffix('.dng').as_posix(), tmp_path.as_posix())
    os.system(cmd)

    os.remove(tmp_path)


def save_as_tiff(int_im, outpath):
    # 16 bits uncompressed by default
    # Imageio is the only module I could find to save 16 bits RGB tiffs without compression (cv2 does LZW).
    # It is vital to have uncompressed image, because validate_dng cannot work if the tiff is compressed.
    imageio.imwrite(outpath.with_suffix('.tif').as_posix(), int_im, bigtiff=False)
