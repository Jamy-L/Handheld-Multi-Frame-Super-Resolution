# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:34:02 2022

@author: jamyl
"""
import os
import glob
from time import time

import numpy as np
import cv2
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
import exifread
import rawpy
import matplotlib.pyplot as plt

from handheld_super_resolution import process
from evaluation import warp_flow, upscale_alignement

def gamma(image):
    return image**(1/2.2)

def flat(x):
    return 1-np.exp(-x/1000)


#%%

# Warning : tileSize is expressed at grey pixel scale.
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
                'mode':'bayer',
                'epsilon div' : 1e-6,
                'tuning' : {
                    'tileSizes' : 16,
                    'kanadeIter': 6, # 3 
                    }},
            'robustness' : {
                'mode':'bayer',
                'tuning' : {
                    'tileSizes': 16,
                    't' : 0,            # 0.12
                    's1' : 2,          #12
                    's2' : 12,              # 2
                    'Mt' : 0.8,         # 0.8
                    }
                },
            'merging': {
                'mode':'bayer',
                'scale': 2,
                'tuning': {
                    'tileSizes': 16,
                    'k_detail' : 0.25, # [0.25, ..., 0.33]
                    'k_denoise': 5,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    }
                }}

options = {'verbose' : 3}
burst_path = 'P:/0001/Samsung'

output, R, r, alignment = process(burst_path, options, params)


#%%

raw_ref_img = rawpy.imread(burst_path + '/im_00.dng')
exif_tags = open(burst_path + '/im_00.dng', 'rb')
tags = exifread.process_file(exif_tags)
ref_img = raw_ref_img.raw_image.copy()


comp_images = rawpy.imread(
    burst_path + '/im_01.dng').raw_image.copy()[None]
for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('{}/im_0{}.dng'.format(burst_path, i)
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
plt.imshow(gamma(output_img))
base = np.empty((int(ref_img.shape[0]/2), int(ref_img.shape[1]/2), 3))
base[:,:,0] = ref_img[0::2, 1::2]
base[:,:,1] = (ref_img[::2, ::2] + ref_img[1::2, 1::2])/2
base[:,:,2] = ref_img[1::2, ::2]

plt.figure("original bicubic")
plt.imshow(cv2.resize(gamma(base/1023), None, fx = 2*2, fy = 2*2, interpolation=cv2.INTER_CUBIC))

#%% warp optical flow

ref_grey_image = (ref_img[::2, ::2] + ref_img[1::2, ::2] +
                  ref_img[::2, 1::2] + ref_img[1::2, 1::2])/4
plt.figure('ref grey')    
plt.imshow(ref_grey_image, cmap= 'gray')  

comp_grey_images = (comp_images[:, ::2, ::2] + comp_images[:,1::2, ::2] +
                    comp_images[:,::2, 1::2] + comp_images[:,1::2, 1::2])/4
grey_al = alignment.copy()
grey_al[:,:,:,-2:]*=0.5 # pure translation are downgraded from bayer to grey
upscaled_al = upscale_alignement(grey_al, ref_grey_image.shape[:2], 16, v2=True) # half tile size cause grey
for image_index in range(comp_images.shape[0]):
    warped = warp_flow(comp_grey_images[image_index]/1023, upscaled_al[image_index], rgb = False)
    plt.figure("image {}".format(image_index))
    plt.imshow(warped, cmap = 'gray')
    plt.figure("EQ {}".format(image_index))
    plt.imshow(np.log10((ref_grey_image/1023 - warped)**2),vmin = -4, vmax =0 , cmap="Reds")
    plt.colorbar()
    
    print("Im {}, EQM = {}".format(image_index, np.mean((ref_grey_image/1023 - warped)**2)))
    
  

#%% histograms
r2 = np.mean(R, axis = 3)
plt.figure("R histogram")
plt.hist(r2.reshape(r2.size), bins=25)
r3 = np.mean(r, axis = 3)
plt.figure("r histogram")
plt.hist(r3.reshape(r3.size), bins=25)
#%% kernels
def plot_merge(cov_i, D, pos):
    cov_i = cov_i[pos]
    D = D[pos]
    L = np.linspace(-5, 5, 100)
    Xm, Ym = np.meshgrid(L,L)
    Z = np.empty_like(Xm)
    Z = cov_i[0,0] * Xm**2 + (cov_i[1, 0] + cov_i[0, 1])*Xm*Ym + cov_i[1, 1]*Ym**2
    plt.figure("merge in (x = {}, y = {})".format(pos[1],pos[0]))
    plt.pcolor(Xm, Ym, np.exp(-Z/2), vmin = 0, vmax=1)
    plt.gca().invert_yaxis()
    plt.scatter([0], [0], c='g', marker ='o')
    plt.scatter(D[:,:,0].reshape(9), D[:,:,1].reshape(9), c='r', marker ='x')

    # for i in range(9):
    #     plt.quiver(D[:,:,0].reshape(9)[i], -D[:,:,1].reshape(9)[i], scale=1, scale_units = "xy")
    plt.colorbar()

#%% robustness
plt.figure('r')
plt.imshow(r[0]/np.max(r[0], axis=(0,1)))
plt.figure("accumulated r")
plt.imshow(np.mean(r[0]/np.max(r[0], axis=(0,1)), axis = 2), cmap = "gray")

plt.figure('R')
plt.imshow(R[0]/np.max(R[0], axis=(0,1)))


#%% eighen vectors
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

