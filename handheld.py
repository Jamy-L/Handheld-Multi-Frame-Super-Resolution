# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:34:02 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow_V2, get_closest_flow_V2
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime

from linalg import quad_mat_prod
from robustness import fetch_robustness, compute_robustness
from merge import merge
from block_matching import alignHdrplus
from kernels import plot_kernel

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
import exifread
import os
import glob


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
    
    
    cuda_final_alignment = lucas_kanade_optical_flow_V2(
        ref_img, comp_imgs, pre_alignment, options, params['kanade'])

    current_time = time()
    cuda_Robustness, cuda_robustness = compute_robustness(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment,
                                         options, params['merging'])

    current_time = getTime(
        current_time, 'Robustness estimated')
    
    output = merge(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment, cuda_robustness, {"verbose": 3}, params['merging'])
    print('\nTotal ellapsed time : ', time() - t1)
    return output, cuda_Robustness.copy_to_host(), cuda_robustness.copy_to_host()
    
#%%

def process(burst_path, options, params):
    currentTime, verbose = time(), options['verbose'] > 1
    

    ref_id = 0 #TODO get ref id
    
    raw_comp = []
    
    # Get the list of raw images in the burst path
    raw_path_list = glob.glob(os.path.join(burst_path, '*.dng'))
    assert raw_path_list != [], 'At least one raw .dng file must be present in the burst folder.'
	# Read the raw bayer data from the DNG files
    for index, raw_path in enumerate(raw_path_list):
        with rawpy.imread(raw_path) as rawObject:
            if index != ref_id :
                
                raw_comp.append(rawObject.raw_image.copy())  # copy otherwise image data is lost when the rawpy object is closed
         
    # Reference image selection and metadata         
    ref_raw = rawpy.imread(raw_path_list[ref_id]).raw_image.copy()
    with open(raw_path_list[ref_id], 'rb') as raw_file:
        tags = exifread.process_file(raw_file)
    
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
        
    params['merging']['exif']['white level'] = str(tags['Image Tag 0xC61D'])
    CFA = str((tags['Image CFAPattern']))[1:-1].split(sep=', ')
    CFA = np.array([int(x) for x in CFA]).reshape(2,2)
    params['merging']['exif']['CFA Pattern'] = CFA
    
    if verbose:
        currentTime = getTime(currentTime, ' -- Read raw files')

    return main(ref_raw, np.array(raw_comp), options, params)

#%%
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
                'epsilon div' : 1e-6,
                'tuning' : {
                    'tileSizes' : 32,
                    'kanadeIter': 6, # 3 
                    }},
            'merging': {
                'scale': 1,
                'tuning': {
                    'tileSizes': 32,
                    'k_detail' : 0.33, # [0.25, ..., 0.33]
                    'k_denoise': 5,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    't' : 0,            # 0.12
                    's1' : 2,          #12
                    's2' : 12,              # 2
                    'Mt' : 0.8,         # 0.8
                    'sigma_t' : 30,
                    'dt' : 1e-3},
                    }
            }

options = {'verbose' : 3}
burst_path = 'P:/0001/Samsung'

output, R, r = process(burst_path, options, params)


#%%

raw_ref_img = rawpy.imread('P:/0001/Samsung/im_00.dng')
exif_tags = open('P:/0001/Samsung/im_00.dng', 'rb')
tags = exifread.process_file(exif_tags)
ref_img = raw_ref_img.raw_image.copy()


comp_images = rawpy.imread(
    'P:/0001/Samsung/im_01.dng').raw_image.copy()[None]
for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('P:/0001/Samsung/im_0{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)





output_img = output[:,:,:3].copy()
imsize = output_img.shape

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

covs = np.empty((output.shape[0], output.shape[1], 2, 2))
covs[:,:,0,0] = output[:,:,9].copy()
covs[:,:,0,1] = output[:,:,10].copy()
covs[:,:,1,0] = output[:,:,11].copy()
covs[:,:,1,1] = output[:,:,12].copy()

D = np.empty((output.shape[0], output.shape[1], 3, 3, 2))
for i in range(18):
    D[:, :, i//6, (i%6)//2, i%2] = output[:,:,13 + i].copy() 

print('Nan detected in output: ', np.sum(np.isnan(output_img)))
print('Inf detected in output: ', np.sum(np.isinf(output_img)))

plt.figure("output")
plt.imshow(gamma(output_img/1023))
base = np.empty((int(ref_img.shape[0]/2), int(ref_img.shape[1]/2), 3))
base[:,:,0] = ref_img[0::2, 1::2]
base[:,:,1] = (ref_img[::2, ::2] + ref_img[1::2, 1::2])/2
base[:,:,2] = ref_img[1::2, ::2]

plt.figure("original bicubic")
plt.imshow(cv2.resize(gamma(base/1023), None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC))


r2 = np.mean(R, axis = 3)
plt.figure("R histogram")
plt.hist(r2.reshape(r2.size), bins=25)
r3 = np.mean(r, axis = 3)
plt.figure("r histogram")
plt.hist(r3.reshape(r3.size), bins=25)
#%%
def plot_merge(cov_i, D, pos):
    cov_i = cov_i[pos]
    D = D[pos]
    L = np.linspace(-10, 10, 100)
    Xm, Ym = np.meshgrid(L,L)
    Z = np.empty_like(Xm)
    Z = cov_i[0,0] * Xm**2 + (cov_i[1, 0] + cov_i[0, 1])*Xm*Ym + cov_i[1, 1]*Ym**2
    plt.figure()
    plt.pcolor(Xm, Ym, np.exp(-Z/2), vmin = 0, vmax=1)
    plt.gca().invert_yaxis()
    plt.scatter(D[:,:,0].reshape(9), D[:,:,1].reshape(9), c='r', marker ='x')
    # for i in range(9):
    #     plt.quiver(D[:,:,0].reshape(9)[i], -D[:,:,1].reshape(9)[i], scale=1, scale_units = "xy")
    plt.colorbar()

plt.figure('r')
plt.imshow(r[0]/np.max(r[0], axis=(0,1)))
plt.figure("accumulated r")
plt.imshow(np.mean(r[0]/np.max(r[0], axis=(0,1)), axis = 2), cmap = "gray")

plt.figure('R')
plt.imshow(R[0]/np.max(R[0], axis=(0,1)))


# # quivers for eighenvectors
# # Lower res because pyplot's quiver is really not made for that (=slow)
# plt.figure('quiver')
# scale = 5*1e1
# downscale_coef = 4
# ix, iy = 2, 1
# patchx, patchy = int(imsize[1]/downscale_coef), int(imsize[0]/downscale_coef)
# plt.imshow(gamma(output_img[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1)]/1023))

# # minus sign because pyplot takes y axis growing towards top. but we need the opposite
# plt.quiver(e1[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 0],
#            -e1[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 1], width=0.001,linewidth=0.0001, scale=scale)
# plt.quiver(e2[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 0],
#            -e2[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 1], width=0.001,linewidth=0.0001, scale=scale, color='b')


# img = process_isp(raw=raw_ref_img, img=(output_img/1023), do_color_correction=False, do_tonemapping=True, do_gamma=True, do_sharpening=False)

