# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:59:41 2022

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
from skimage.transform import warp 
import colour_demosaicing

from handheld_super_resolution.super_resolution import main
from handheld_super_resolution.block_matching import alignBurst
from handheld_super_resolution.optical_flow import get_closest_flow, lucas_kanade_optical_flow


burst_path = 'P:/aether-4K/aether-4K_defense'
# burst_path = 'P:/aether-4K/aether-4K_beach'


raw_path_list = glob.glob(os.path.join(burst_path, '*.tiff'))



first_image_path = raw_path_list[0]

raw_ref_img =  np.array(Image.open(first_image_path))[::2, ::2]



comp_images = np.array(Image.open(raw_path_list[1]))[::2,::2][None]
for i in range(2,len(raw_path_list)):
    comp_images = np.append(comp_images, np.array(Image.open(raw_path_list[i]
                                                      ))[::2, ::2][None], axis=0)

maxi = max(np.max(raw_ref_img), np.max(comp_images))
mini = min(np.min(raw_ref_img), np.max(comp_images))

raw_ref_img = (raw_ref_img - mini)/(maxi - mini)
comp_images = (comp_images - mini)/(maxi - mini)
# comp_images = comp_images[:20]


#%% params
CFA = np.array([[2, 1], [1, 0]]) # dummy cfa

params = {'block matching': {
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
                'mode':'gray',
                'epsilon div' : 1e-6,
                'tuning' : {
                    'tileSize' : 16,
                    'kanadeIter': 6, # 3 
                    }},
            'robustness' : {
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
options = {'verbose' : 3}

#%%

output, R, r, alignment = main(np.ascontiguousarray(raw_ref_img),
                               np.ascontiguousarray(comp_images),
                               options, params)
#%% show output

plt.figure("output 2")
plt.imshow(output[:,:,0], cmap="gray")


#%% plot burst
plt.figure("ref image")
plt.imshow(raw_ref_img, cmap="gray")



for image_index in range(comp_images.shape[0]):
    plt.figure("image {}".format(image_index))
    plt.imshow(comp_images[image_index], cmap="gray")


