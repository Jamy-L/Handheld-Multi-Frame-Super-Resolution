# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:44:36 2023

@author: jamyl
"""

import os

import numpy as np
import exifread
import rawpy
import imageio

EXIFTOOL_PATH = 'C:/Users/jamyl/Downloads/exiftool-12.44/exiftool.exe'
DNG_VALIDATE_PATH = 'C:/Users/jamyl/Downloads/dng_sdk_1_6/dng_sdk_1_6/dng_sdk/targets/win/debug64_x64/dng_validate.exe'

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
    
    
    np_int_img = np.copy(np_img) # copying to avoid inplace-overwritting
    # Undo White balance and black level
    ## get tags
    with open(ref_dng_path, 'rb') as raw_file:
        tags = exifread.process_file(raw_file)

    white_level = tags['Image Tag 0xC61D'].values[0] # there is only one white level
    
    black_levels = tags['Image BlackLevel'] 
    if isinstance(black_levels.values[0], int):
        black_levels = np.array(black_levels.values)
    else: # Sometimes this tag is a fraction object for some reason. It seems that black levels are all integers anyway
        black_levels = np.array([int(x.decimal()) for x in black_levels.values])
    
    raw = rawpy.imread(ref_dng_path)
    white_balance = raw.camera_whitebalance
    
    ## Reverse WB
    
    for c in range(3):
        np_int_img[:, :, c] /= white_balance[c] / white_balance[1]
        np_int_img[:, :, c] = np_int_img[:, :, c] * (white_level - black_levels[c]) + black_levels[c]

    np_int_img = np.clip(np_int_img, 0, 2**16-1).astype(np.uint16)
    
    # Saving the image as 16 bits RGB tiff 
    save_as_tiff(np_int_img, outpath)
    
    tmp_path = outpath.parent/'tmp.dng'
    
    # Deleting tmp.dng if it is already existing
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    
    
    # Overwritting the tiff tags with dng tags, and replacing the .tif extension
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

