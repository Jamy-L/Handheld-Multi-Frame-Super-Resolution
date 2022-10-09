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
import colour_demosaicing
# pip install colour-demosaicing

from handheld_super_resolution import process, raw2rgb
from evaluation import warp_flow, upscale_alignement



def flat(x):
    return 1-np.exp(-x/1000)


def cfa_to_grayscale(raw_img):
    return (raw_img[..., 0::2, 0::2] + 
            raw_img[..., 1::2, 0::2] + 
            raw_img[..., 0::2, 1::2] + 
            raw_img[..., 1::2, 1::2])/4


#%%

# Warning : tileSize is expressed at grey pixel scale.
params = {'scale' : 2,
          'mode' : 'bayer',
          'block matching': {
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
                    'tileSize' : 8,
                    # 'tileSizes' : 8,
                    'kanadeIter': 6, # 3
                    }},
            'robustness' : {
                'tuning' : {
                    't' : 0.12,            # 0.12
                    's1' : 12,           # 12
                    's2' : 2,          # 2
                    'Mt' : 0.8,         # 0.8
                    }
                },
            'merging': {
                'tuning': {
                    'k_detail' : 0.25, # [0.25, ..., 0.33]
                    'k_denoise': 5,    # [3.0, ...,5.0]
                    'D_th': 0.05,      # [0.001, ..., 0.010]
                    'D_tr': 0.014,     # [0.006, ..., 0.020]
                    'k_stretch' : 4,   # 4
                    'k_shrink' : 2,    # 2
                    }
                }}

options = {'verbose' : 3}
burst_path = 'P:/0002/Samsung'

output, R, r, alignment = process(burst_path, options, params)


#%% exttracting images locally for comparison 

first_image_path = os.path.join(burst_path, 'im_00.dng')
raw_ref_img = rawpy.imread(first_image_path)
exif_tags = open(first_image_path, 'rb')
tags = exifread.process_file(exif_tags)
ref_img = raw_ref_img.raw_image.copy()

xyz2cam = raw2rgb.get_xyz2cam_from_exif(first_image_path)

comp_images = rawpy.imread(
    burst_path + '/im_01.dng').raw_image.copy()[None]
for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('{}/im_0{}.dng'.format(burst_path, i)
                                                      ).raw_image.copy()[None], axis=0)

white_level = tags['Image Tag 0xC61D'].values[0] # there is only one white level

black_levels = tags['Image BlackLevel'] # This tag is a fraction for some reason. It seems that black levels are all integers anyway
black_levels = np.array([int(x.decimal()) for x in black_levels.values])

white_balance = raw_ref_img.camera_whitebalance

CFA = tags['Image CFAPattern']
CFA = np.array([x for x in CFA.values]).reshape(2,2)

ref_img = ref_img.astype(np.float32)
comp_images = comp_images.astype(np.float32)
for i in range(2):
    for j in range(2):
        channel = channel = CFA[i, j]
        ref_img[i::2, j::2] = (ref_img[i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
        ref_img[i::2, j::2] *= white_balance[channel]
        
        comp_images[:, i::2, j::2] = (comp_images[:, i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
        comp_images[:, i::2, j::2] *= white_balance[channel]



ref_img = np.clip(ref_img, 0.0, 1.0)
comp_images = np.clip(comp_images, 0.0, 1.0)


#%% extracting handhled's output data


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
plt.imshow(raw2rgb.postprocess(raw_ref_img, output_img, xyz2cam=xyz2cam))

base = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(ref_img, pattern='GRBG')  # pattern is [[G,R], [B,G]] for the Samsung G8


plt.figure("original bicubic")
plt.imshow(cv2.resize(raw2rgb.postprocess(raw_ref_img, base, xyz2cam=xyz2cam), None, fx = params["merging"]['scale'], fy = params["merging"]['scale'], interpolation=cv2.INTER_CUBIC))

#%% warp optical flow

plt.figure('ref grey')    
ref_grey_image = cfa_to_grayscale(ref_img)
plt.imshow(ref_grey_image, cmap= 'gray')  

comp_grey_images = cfa_to_grayscale(comp_images)
grey_al = alignment.copy()
grey_al[:,:,:,-2:]*=0.5 # pure translation are downscaled from bayer to grey
upscaled_al = upscale_alignement(grey_al, ref_grey_image.shape[:2], 8) # half tile size cause grey
for image_index in range(comp_images.shape[0]):
    warped = warp_flow(comp_grey_images[image_index], upscaled_al[image_index], rgb = False)
    plt.figure("image {}".format(image_index))
    plt.imshow(warped, cmap = 'gray')
    plt.figure("EQ {}".format(image_index))
    plt.imshow(np.log10((ref_grey_image - warped)**2),vmin = -4, vmax =0 , cmap="Reds")
    plt.colorbar()
    
    print("Im {}, EQM = {}".format(image_index, np.mean((ref_grey_image - warped)**2)))
    
  

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
# plt.imshow(raw2rgb.postprocess(output_img[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1)]/1023))

# # minus sign because pyplot takes y axis growing towards top. but we need the opposite
# plt.quiver(e1[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 0],
#            -e1[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 1], width=0.001,linewidth=0.0001, scale=scale)
# plt.quiver(e2[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 0],
#            -e2[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 1], width=0.001,linewidth=0.0001, scale=scale, color='b')


# img = process_isp(raw=raw_ref_img, img=(output_img/1023), do_color_correction=False, do_tonemapping=True, do_gamma=True, do_sharpening=False)

