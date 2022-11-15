# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:46:40 2022

@author: jamyl
"""

from tqdm import tqdm
import random
import glob
import os

import numpy as np
from numba import cuda, float64
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters
from skimage.transform import warp 
import colour_demosaicing

from plot_flow import flow2img

from evaluation import upscale_alignement
from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.block_matching import alignBurst
from handheld_super_resolution.optical_flow import get_closest_flow, lucas_kanade_optical_flow


#%% params
CFA = np.array([[2, 1], [1, 0]]) # dummy cfa

params = {'mode':'grey',
          'block matching': {
                'grey method':'f',
                'mode':'gray',
                'tuning': {
                    # WARNING: these parameters are defined fine-to-coarse!
                    'factors': [1, 2, 2, 2],
                    'tileSizes': [16, 16, 16, 8],
                    'searchRadia': [1, 4, 4, 4],
                    'distances': ['L1', 'L2', 'L2', 'L2'],
                    # if you want to compute subpixel tile alignment at each pyramid level
                    'subpixels': [False, True, True, True]
                    }},
            'kanade' : {
                'grey method':'f',
                'mode':'gray',
                'epsilon div' : 1e-6,
                'grey method':'FFT',
                'tuning' : {
                    'tileSize' : 16,
                    'kanadeIter': 6, # 3 
                    'sigma blur':0,
                    }},
            'robustness' : {
                'on':False,
                'exif':{'CFA Pattern':CFA},
                'mode':'gray',
                'tuning' : {
                    'tileSize': 16,
                    't' : 0.12,            # 0.12
                    's1' : 12,          #12
                    's2' : 2,           # 2
                    'Mt' : 0.8,         # 0.8
                    }
                },
            'merging': {
                'exif':{'CFA Pattern':CFA},
                'mode':'gray',
                'scale': 2,
                'kernel' : 'handheld',
                'tuning': {
                    'tileSize': 16,
                    'k_detail' : 0.33, # [0.25, ..., 0.33]
                    'k_denoise': 5,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    }
                }}
params['robustness']['std_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_std_ISO_50.npy')
params['robustness']['diff_curve'] = np.load('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/data/noise_model_diff_ISO_50.npy')
options = {'verbose' : 0}

#%%
burst_path = 'P:/results_surecavi_20220325/results_anger/20220325/data/exp2-stack'
output_path = 'P:/results_surecavi_20220325/results_anger/20220325/results_handheld/exp2'

def process_burst(ref_id, burst_size):
    raw_path_list = glob.glob(os.path.join(burst_path, '*.png'))


    index_start = ref_id
    first_image_path = raw_path_list[index_start]

    raw_ref_img = cv2.imread(first_image_path, -1)

    comp_images = cv2.imread(raw_path_list[index_start+1], -1)[None]
    for i in range(index_start+2,index_start+burst_size+1):
        comp_images = np.append(comp_images, cv2.imread(raw_path_list[i], -1)[None], axis=0)

    raw_ref_img = raw_ref_img/(2**16 - 1)
    comp_images = comp_images/(2**16 - 1)

    output, R, r, alignment, covs = main(np.ascontiguousarray(raw_ref_img).astype(np.float32),
                                         np.ascontiguousarray(comp_images).astype(np.float32),
                                         options, params)
    name = "{}/{}-N{}.tiff".format(output_path, ref_id, burst_size)
    cv2.imwrite(name,filters.unsharp_mask(output[:,:,0], radius=3, amount=1.5,
                               channel_axis=None, preserve_range=True))
for i in tqdm(range(40, 89)):
    process_burst(i, 11)