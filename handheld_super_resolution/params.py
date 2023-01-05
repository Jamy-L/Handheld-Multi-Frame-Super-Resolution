# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:22:54 2022

This scripts aims to fetch the appropriate parameters, based on the estimated
SNR.

@author: jamyl
"""
import numpy as np
import warnings

def get_params(SNR):
    SNR = np.clip(SNR, 6, 30)
    if SNR <= 14:
        Ts = 64
    elif SNR <= 22:
        Ts = 32
    else:
        Ts = 16
        
    
    params = {'scale' : 1,
              'mode' : 'bayer', # 'bayer' or 'grey' 
              'grey method' : 'FFT',
              'debug': False, # when True, a dict is returned with debug infos.
              'block matching': {
                    'tuning': {
                        # WARNING: these parameters are defined fine-to-coarse!
                        'factors': [1, 2, 4, 4],
                        'tileSizes': [Ts, Ts, Ts, Ts//2],
                        'searchRadia': [1, 4, 4, 4],
                        'distances': ['L1', 'L2', 'L2', 'L2'],
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
                        't' : 0.12,       # 0.12
                        's1' : 2,         # 2
                        's2' : 12,        # 12
                        'Mt' : 0.8,       # 0.8
                        }
                    },
                'merging': {
                    'kernel' : 'handheld', # 'act' for act kernel, 'handheld' for handhel kernel
                    'tuning': {
                        'k_detail' : 0.25 + (0.33 - 0.25)*(30 - SNR)/(30 - 6), # [0.25, ..., 0.33]
                        'k_denoise': 3 + (5 - 3)*(30 - SNR)/(30 - 6),    # [3.0, ...,5.0]
                        'D_th': 0.001 + (0.01 - 0.001)*(30 - SNR)/(30 - 6),      # [0.001, ..., 0.010]
                        'D_tr': 0.006 + (0.02*16 - 0.006)*(30 - SNR)/(30 - 6),     # [0.006, ..., 0.020]
                        'k_stretch' : 4,   # 4
                        'k_shrink' : 2,    # 2
                        }
                    },
                
                'accumulated robustness denoiser' : {
                    'on' : True,
                    'sigma max' : 1.5, # std of the gaussian blur applied when only 1 frame is merged
                    'max frame count' : 8 # number of merged frames above which no blurr is applied
                    },
                'post processing' : {
                    'on':True,
                    'do color correction':True,
                    'do tonemapping':True,
                    'do gamma' : True,
                    'do sharpening' : True,
                    
                    'sharpening':{
                        'radius':3,
                        'ammount':1.5
                        }
                    }
                }

    return params

def check_params_validity(params, imshape):
    if params["grey method"] != "FFT":
        raise NotImplementedError("Grey level images should be obtained with FFT")
        
    assert params['scale'] >= 1
    if params['scale'] > 3:
        warnings.warn("Warning.... The required scale is superior to 3, but the algorighm can hardly go above.")
    
    if (not params['robustness']['on']) and params['accumulated robustness denoiser']['on']:
        warnings.warn("Warning.... Robustness based denoising is enabled, "
                      "but robustness is disabled. No further denoising will be done.")
        
    assert params['merging']['kernel'] in ['handheld', 'act']
    assert params['mode'] in ["bayer", 'grey']
    assert params['kanade']['tuning']['kanadeIter'] > 0
    assert params['kanade']['tuning']['sigma blur'] >= 0
    
    assert len(imshape) == 2
    
    Ts = params['block matching']['tuning']['tileSizes'][0]
    
    # Checking if block matching is possible
    padded_imshape_x = Ts*(int(np.ceil(imshape[1]/Ts)))
    padded_imshape_y = Ts*(int(np.ceil(imshape[0]/Ts)))
    
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
    
def merge_params(dominant, recessive):
    """
    Merges 2 sets of parameters, one being dominant (= overwrittes the recessive
                                                     when a value a specified)
    """
    recessive_ = recessive.copy()
    for dom_key in dominant.keys():
        if (dom_key in recessive_.keys()) and type(dominant[dom_key]) is dict:
            recessive_[dom_key] = merge_params(dominant[dom_key], recessive_[dom_key])
        else:
            recessive_[dom_key] = dominant[dom_key]
    return recessive_

    