# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:22:54 2022

This scripts aims to fetch the appropriate parameters, based on the estimated
PSNR.

@author: jamyl
"""
import numpy as np

def get_params(PSNR):
    # TODO compute PSNR
    PSNR = np.clip(PSNR, 6, 30)
    if PSNR <= 14:
        Ts = 64
    elif PSNR <= 22:
        Ts = 32
    else:
        Ts = 16
        
    
    params = {'scale' : 1,
              'mode' : 'bayer', # 'bayer' or grey if something else 
              'grey method' : 'FFT',
              'debug': False, # when True, a dict is returned with tensors.
              'block matching': {
                    'tuning': {
                        # WARNING: these parameters are defined fine-to-coarse!
                        'factors': [1, 2, 4, 4],
                        'tileSizes': [Ts, Ts, Ts, int(Ts/2)],
                        'searchRadia': [1, 4, 4, 4],
                        'distances': ['L1', 'L2', 'L2', 'L2'],
                        # if you want to compute subpixel tile alignment at each pyramid level
                        'subpixels': [False, True, True, True]
                        }},
                'kanade' : {
                    'tuning' : {
                        'kanadeIter': 3, # 3
                        # gaussian blur before computing grads. If 0, no blur is applied
                        'sigma blur':0.5,
                        }},
                'robustness' : {
                    'on':True,
                    'tuning' : {
                        't' : 0.12,         # 0.12
                        's1' : 2,           # 12
                        's2' : 12,#237,         # 2
                        'Mt' : 0.8,         # 0.8
                        }
                    },
                'merging': {
                    'kernel' : 'handheld', # 'act' for act kernel, other for handhel kernel
                    'tuning': {
                        'k_detail' : 0.25 + (0.33 - 0.25)*(30 - PSNR)/(30 - 6), # [0.25, ..., 0.33]
                        'k_denoise': 3 + (5 - 3)*(30 - PSNR)/(30 - 6),    # [3.0, ...,5.0]
                        'D_th': 0.001 + (0.01 - 0.001)*(30 - PSNR)/(30 - 6),      # [0.001, ..., 0.010]
                        'D_tr': 0.006 + (0.02*16 - 0.006)*(30 - PSNR)/(30 - 6),     # [0.006, ..., 0.020]
                        'k_stretch' : 4,   # 4
                        'k_shrink' : 2,    # 2
                        }
                    }}
    return params

def check_params_validity(params, imshape):
    if params["grey method"] != "FFT":
        raise NotImplementedError("Grey level images should be obtained with FFT")
        
    assert params['scale'] >= 1
    assert params['merging']['kernel'] in ['handheld', 'act']
    assert params['mode'] in ["bayer", 'grey']
    assert params['kanade']['tuning']['kanadeIter'] > 0
    assert params['kanade']['tuning']['sigma blur'] >= 0
    
    assert len(imshape) == 2
    
    Ts = params['block matching']['tuning']['tileSizes'][0]
    
    # Checking if block matching is possible
    padded_imshape_x = Ts*(int(np.ceil(imshape[1]/Ts)) + 1)
    padded_imshape_y = Ts*(int(np.ceil(imshape[0]/Ts)) + 1)
    
    lvl_imshape_y, lvl_imshape_x = padded_imshape_y, padded_imshape_x
    for lvl, (factor, ts) in enumerate(zip(params['block matching']['tuning']['factors'], params['block matching']['tuning']['tileSizes'])):
        lvl_imshape_y, lvl_imshape_x = np.floor(lvl_imshape_y/factor), np.floor(lvl_imshape_x/factor)
        
        n_tiles_y = lvl_imshape_y/ts
        n_tiles_x = lvl_imshape_x/ts
        
        if n_tiles_y < 1 or n_tiles_x < 1:
            raise ValueError("Image of shape {} is incompatible with the given "\
                             "block matching tile sizes and factors : at level {}, "\
                             "coarse image of shape {} cannot be divided into "\
                             "tiles of size {}.".format(
                                 imshape, lvl,
                                 (lvl_imshape_y, lvl_imshape_x),
                                 ts))
    


    