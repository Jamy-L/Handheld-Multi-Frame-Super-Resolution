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
              'block matching': {
                    'tuning': {
                        # WARNING: these parameters are defined fine-to-coarse!
                        'factors': [1, 2, 2, 4],
                        'tileSizes': [Ts, Ts, Ts, int(Ts/2)],
                        'searchRadia': [1, 4, 4, 4],
                        'distances': ['L1', 'L2', 'L2', 'L2'],
                        # if you want to compute subpixel tile alignment at each pyramid level
                        'subpixels': [False, True, True, True]
                        }},
                'kanade' : {
                    # FFT, demosaicing, decimating
                    'grey method':"FFT",
                    'tuning' : {
                        'kanadeIter': 6, # 3
                        # gaussian blur before computing grads. If 0, no blur is applied
                        'sigma blur':0,
                        }},
                'robustness' : {
                    'on':True,
                    'tuning' : {
                        't' : 0.12,         # 0.12
                        's1' : 2,           # 12
                        's2' : 237,         # 2
                        'Mt' : 0.8,         # 0.8
                        }
                    },
                'merging': {
                    'kernel' : 'handheld', # 'act' for act kernel, other for handhel kernel
                    'tuning': {
                        'k_detail' : 0.25 + (0.33 - 0.25)*(30 - PSNR)/(30 - 6), # [0.25, ..., 0.33]
                        'k_denoise': 3 + (5 - 3)*(30 - PSNR)/(30 - 6),    # [3.0, ...,5.0]
                        'D_th': 0.001 + (0.01 - 0.001)*(30 - PSNR)/(30 - 6),      # [0.001, ..., 0.010]
                        'D_tr': 0.006 + (0.02 - 0.006)*(30 - PSNR)/(30 - 6),     # [0.006, ..., 0.020]
                        'k_stretch' : 4,   # 4
                        'k_shrink' : 2,    # 2
                        }
                    }}
    return params