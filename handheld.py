# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:34:02 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow, get_closest_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_kernel_cov
from linalg import quad_mat_prod
from robustness import fetch_robustness, compute_robustness
from merge import merge
from block_matching import alignHdrplus

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
from time import time
import cupy as cp
from scipy.interpolate import interp2d
import math
from torch import from_numpy

from fast_two_stage_psf_correction.fast_optics_correction.raw2rgb import process_isp




def gamma(image):
    return image**(1/2.2)

def flat(x):
    return 1-np.exp(-x/1000)


def main(ref_img, comp_imgs, options, params):
    t1 = time()
    
    pre_alignment, aligned_tiles = alignHdrplus(ref_img, comp_imgs,params['block matching'], options)
    pre_alignment = pre_alignment[:, :, :, ::-1] # swapping x and y direction (x must be first)
    
    current_time = time()
    cuda_ref_img = cuda.to_device(ref_img)
    cuda_comp_imgs = cuda.to_device(comp_imgs)
    current_time = getTime(
        current_time, 'Arrays moved to GPU')
    
    
    cuda_final_alignment = lucas_kanade_optical_flow(
        ref_img, comp_imgs, pre_alignment, options, params['kanade'])

    current_time = time()
    cuda_robustness = compute_robustness(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment,
                                         options, params['merging'])
    
    current_time = getTime(
        current_time, 'Robustness estimated')
    
    output = merge(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment, cuda_robustness, {"verbose": 3}, params['merging'])
    print('\nTotal ellapsed time : ', time() - t1)
    return output, cuda_robustness.copy_to_host()
    


raw_ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N000.dng'
                        )
ref_img = raw_ref_img.raw_image.copy()


comp_images = rawpy.imread(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)


params = {'block matching': {
                'mode':'bayer',
                'tuning': {
                    # WARNING: these parameters are defined fine-to-coarse!
                    'factors': [1, 2, 4, 4],
                    'tileSizes': [16, 16, 16, 8],
                    'searchRadia': [1, 4, 4, 4],
                    'distances': ['L1', 'L2', 'L2', 'L2'],
                    # if you want to compute subpixel tile alignment at each pyramid level
                    'subpixels': [False, True, True, True]
                    }},
            'kanade' : {
                'tuning' : {
                    'tileSizes' : 32,
                    'kanadeIter': 3, # 3 
                    }},
            'merging': {
                'scale': 1,
                'tuning': {
                    'tileSizes': 32,
                    'k_detail' : 0.3,  # [0.25, ..., 0.33]
                    'k_denoise': 4,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    't' : 0.12,        # 0.12
                    's1' : 2,
                    's2' : 2,
                    'Mt' : 0.8,
                    'sigma_t' : 2,
                    'dt' : 15},
                    }
            }

options = {'verbose' : 3}
output, r = main(ref_img, comp_images[:-1], options, params)


#%%
output_img = output[:,:,:3].copy()

l1 = output[:,:,7].copy()
l2 = output[:,:,8].copy()

e1 = np.empty((output.shape[0], output.shape[1], 2))
e1[:,:,0] = output[:,:,3].copy()
e1[:,:,1] = output[:,:,4].copy()
e1[:,:,0]*=flat(l1)
e1[:,:,1]*=flat(l1)


e2 = np.empty((output.shape[0], output.shape[1], 2))
e2[:,:,0] = output[:,:,5].copy()
e2[:,:,1] = output[:,:,6].copy()
e2[:,:,0]*=flat(l2)
e2[:,:,1]*=flat(l2)

print('Nan detected in output: ', np.sum(np.isnan(output_img)))
print('Inf detected in output: ', np.sum(np.isinf(output_img)))

plt.figure("output")
plt.imshow(gamma(output_img/1023))
base = np.empty((int(ref_img.shape[0]/2), int(ref_img.shape[1]/2), 3))
base[:,:,0] = ref_img[1::2, 1::2]
base[:,:,1] = (ref_img[::2, 1::2] + ref_img[1::2, ::2])/2
base[:,:,2] = ref_img[::2, ::2]

plt.figure("original")
plt.imshow(gamma(base/1023))


r2 = np.mean(((r+0.12)/12.12), axis = 3)
plt.figure()
plt.hist(r2.reshape(8*1560*2104))
#%%
# quivers for eighenvectors
# Lower res because pyplot's quiver is really not made for that (=slow)
plt.figure('quiver')
scale = 5*1e1
plt.imshow(gamma(output[:,:,:3][::15, ::15]/1023))
plt.quiver(e1[:,:,0][::15, ::15], e1[:,:,1][::15, ::15], width=0.001,linewidth=0.0001, scale=scale)
plt.quiver(e2[:,:,0][::15, ::15], e2[:,:,1][::15, ::15], width=0.001,linewidth=0.0001, scale=scale, color='b')


# # No RGB matrix for this picture unfortunately... So no color correction
# # img = process_isp(raw=raw_ref_img, img=(output_img/1023), do_color_correction=False, do_tonemapping=True, do_gamma=True, do_sharpening=False)

