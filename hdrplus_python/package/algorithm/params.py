"""Parameters structure for the hdrplus algorithm.
Copyright (c) 2021 Antoine Monod

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

This file implements an algorithm possibly linked to the patent US9077913B2.
This file is made available for the exclusive aim of serving as scientific tool
to verify the soundness and completeness of the algorithm description.
Compilation, execution and redistribution of this file may violate patents rights in certain countries.
The situation being different for every country and changing over time,
it is your responsibility to determine which patent rights restrictions apply to you
before you compile, use, modify, or redistribute this file.
A patent lawyer is qualified to make this determination.
If and only if they don't conflict with any patent terms,
you can benefit from the following license terms attached to this file.
"""

# imports
import rawpy
import datetime


def getParams(mode):
    '''Returns a dictionary of parameters that corresponds to a specific algorithm mode.'''
    params = {}
    if mode == 'full':
        params = {
            'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

            'alignment': {
                'mode': 'bayer',  # images are single image Bayer / Color Filter Arrays
                'tuning': {
                    # WARNING: these parameters are defined fine-to-coarse!
                    'factors': [1, 2, 4, 4],
                    'tileSizes': [16, 16, 16, 8],
                    'searchRadia': [1, 4, 4, 4],
                    'distances': ['L1', 'L2', 'L2', 'L2'],
                    # if you want to compute subpixel tile alignment at each pyramid level
                    'subpixels': [False, True, True, True]
                },
                # rawpy parameters for images with motion fields
                'rawpyArgs': {
                    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
                    'half_size': False,
                    'use_camera_wb': True,
                    'use_auto_wb': False,
                    'no_auto_bright': True,
                    'output_color': rawpy.ColorSpace.sRGB,  # sRGB
                    'output_bps': 8
                },
                'writeMotionFields': False,
                'writeAlignedTiles': False
            },

            'merging': {
                'tuning': {
                    'patchSize': 16,
                    'method': 'DFTWiener',  # 'keepAlternate' / 'pairAverage' / 'DFTWiener'
                    # 'exifNoiseProfile' / 'exifISO' / tuple of (lambdaS, lambdaR) values (sigma = lambdaS*I + lambdaR)
                    'noiseCurve': 'exifNoiseProfile'
                },
                'rawpyArgs': {
                    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
                    'half_size': False,
                    'use_camera_wb': True,
                    'use_auto_wb': False,
                    'no_auto_bright': True,
                    # maybe try rawpy.colorSpace.raw and apply a color matrix in .txt?
                    'output_color': rawpy.ColorSpace.sRGB,
                    # gamma correction not applied by rawpy
                    'gamma': (1, 1),
                    'output_bps': 16
                },
                'writeReferenceImage': False,
                'writeGammaReference': False,
                'writeMergedBayer': False,
                'writeMergedImage': False,
                'writeGammaMerged': False
            },

            'finishing': {
                'tuning': {
                    'ltmGain': 'auto',
                    'gtmContrast': 0.075,
                    'sharpenAmount': [1, 0.5, 0.5],
                    'sharpenSigma': [1, 2, 4],
                    'sharpenThreshold': [0.02, 0.04, 0.06]
                },
                # rawpy parameters for reference and final image
                'rawpyArgs': {
                    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
                    'half_size': False,
                    'use_camera_wb': True,
                    'use_auto_wb': False,
                    'no_auto_bright': True,
                    'output_color': rawpy.ColorSpace.sRGB,
                    # gamma correction not applied by rawpy
                    'gamma': (1, 1),
                    'output_bps': 16
                },
                'writeReferenceImage': False,
                'writeGammaReference': True,
                'writeMergedImage': False,
                'writeGammaMerged': True,
                'writeShortExposure': False,
                'writeLongExposure': False,
                'writeFusedExposure': False,
                'writeLTMImage': False,
                'writeLTMGamma': False,
                'writeGTMImage': False,
                'writeReferenceFinal': True,
                'writeFinalImage': True
            }
        }

    elif mode == 'align':
        params = {
            'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

            'alignment': {
                'mode': 'bayer',  # images are single image Bayer / Color Filter Arrays
                'tuning': {
                    # WARNING: these parameters are defined fine-to-coarse!
                    'factors': [1, 2, 4, 4],
                    'tileSizes': [16, 16, 16, 8],
                    'searchRadia': [1, 4, 4, 4],
                    'distances': ['L1', 'L2', 'L2', 'L2'],
                    # if you want to compute subpixel tile alignment at each pyramid level
                    'subpixels': [False, True, True, True]
                },
                # rawpy parameters for images with motion fields
                'rawpyArgs': {
                    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
                    'half_size': False,
                    'use_camera_wb': True,
                    'use_auto_wb': False,
                    'no_auto_bright': True,
                    'output_color': rawpy.ColorSpace.sRGB,  # sRGB
                    'output_bps': 8
                },
                'writeMotionFields': True,
                'writeAlignedTiles': True
            },

            'merging': None,

            'finishing': None
        }

    elif mode == 'merge':
        params = {
            'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

            'alignment': None,  # assumes burst already aligned

            'merging': {
                'tuning': {
                    'patchSize': 16,
                    'method': 'DFTWiener',  # 'keepAlternate' / 'pairAverage' / 'DFTWiener'
                                # 'exifNoiseProfile' / 'exifISO' / tuple of (lambdaS, lambdaR) values (sigma = lambdaS*I + lambdaR)
                                'noiseCurve': 'exifNoiseProfile'
                },
                'rawpyArgs': {
                    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
                    'half_size': False,
                    'use_camera_wb': True,
                    'use_auto_wb': False,
                    'no_auto_bright': True,
                    # maybe try rawpy.colorSpace.raw and apply a color matrix in .txt?
                    'output_color': rawpy.ColorSpace.sRGB,
                    # gamma correction not applied by rawpy
                    'gamma': (1, 1),
                    'output_bps': 16
                },
                'writeReferenceImage': False,
                'writeGammaReference': False,
                'writeMergedBayer': True,
                'writeMergedImage': False,
                'writeGammaMerged': False
            },

            'finishing': None
        }

    elif mode == 'finish':
        params = {
            'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),

            'alignment': None,  # assumes burst already aligned

            'merging': None,  # assumes burst already merged

            'finishing': {
                'tuning': {
                    'ltmGain': 'auto',
                    'gtmContrast': 0.075,
                                'sharpenAmount': [1, 0.5, 0.5],
                                'sharpenSigma': [1, 2, 4],
                                'sharpenThreshold': [0.02, 0.04, 0.06]
                },
                # rawpy parameters for reference and final image
                'rawpyArgs': {
                    'demosaic_algorithm': rawpy.DemosaicAlgorithm.AHD,  # used in HDR+ supplement
                    'half_size': False,
                    'use_camera_wb': True,
                    'use_auto_wb': False,
                    'no_auto_bright': True,
                    'output_color': rawpy.ColorSpace.sRGB,
                    # gamma correction not applied by rawpy
                    'gamma': (1, 1),
                    'output_bps': 16
                },
                'writeReferenceImage': False,
                'writeGammaReference': False,
                'writeMergedImage': False,
                'writeGammaMerged': False,
                'writeShortExposure': False,
                'writeLongExposure': False,
                'writeFusedExposure': False,
                'writeLTMImage': False,
                'writeLTMGamma': False,
                'writeGTMImage': False,
                'writeReferenceFinal': False,
                'writeFinalImage': True
            }
        }
    return params
