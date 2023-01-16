# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:34:02 2022

@author: jamyl
"""
import os
import glob

# import logging

import numpy as np
from math import sqrt
import cv2
import torch as th
import torch.nn.functional as F
from numba import cuda, float32
import exifread
import rawpy
import matplotlib.pyplot as plt
import colour_demosaicing

from handheld_super_resolution import process, raw2rgb, get_params

from plot_flow import flow2img
from handheld_super_resolution.utils import clamp
from handheld_super_resolution.utils_image import compute_grey_images
from handheld_super_resolution.linalg import get_eighen_elmts_2x2

#%%
options = {'verbose' : 1}

# Overwritting SNR based parameters
params={}
params["scale"] = 1


params['merging'] = {'kernel': 'handheld'}
params['post processing'] = {'on':True,
                    'do sharpening' : True,
                    'do color correction':True,
                    'do tonemapping':True,
                    'do gamma' : True,
                    'do devignette' : False,
        
                    'sharpening' : {'radius':3,
                                    'ammount': 1}
                    }
params['robustness']  = {'on' : True}
# params['accumulated robustness denoiser'] = {'on': True,
#                                              'type': 'merge',
#                                              'sigma max' : 1.5, # std of the gaussian blur applied when only 1 frame is merged
#                                              'max frame count' : 8, # number of merged frames above which no blurr is applied
#                                              'radius max':4
#                                              }
params['debug'] = True

burst_path = 'P:/inriadataset/inriadataset/pixel4a/friant/raw/'
# burst_path = 'P:/inriadataset/inriadataset/pixel3a/rue4/raw'
# burst_path = 'P:/0050/Samsung'

output_img, debug_dict = process(burst_path, options, params)

#%%
print('Nan detected in output: ', np.sum(np.isnan(output_img)))
print('Inf detected in output: ', np.sum(np.isinf(output_img)))

plt.figure("output")

plt.imshow(output_img, interpolation = 'none')
plt.xticks([])
plt.yticks([])


#%% extracting images locally for comparison 

raw_path_list = glob.glob(os.path.join(burst_path, '*.dng'))
first_image_path = raw_path_list[0]

raw_ref_img = rawpy.imread(first_image_path)

exif_tags = open(first_image_path, 'rb')
tags = exifread.process_file(exif_tags)

ref_img = raw_ref_img.raw_image.copy()

xyz2cam = raw2rgb.get_xyz2cam_from_exif(first_image_path)

comp_images = rawpy.imread(
    raw_path_list[1]).raw_image.copy()[None]
for i in range(2,len(raw_path_list)):
    comp_images = np.append(comp_images, rawpy.imread(raw_path_list[i]
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
        ref_img[i::2, j::2] *= white_balance[channel]/ white_balance[1]
        
        comp_images[:, i::2, j::2] = (comp_images[:, i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
        comp_images[:, i::2, j::2] *= white_balance[channel]/ white_balance[1]


ref_img = np.clip(ref_img, 0.0, 1.0)
comp_images = np.clip(comp_images, 0.0, 1.0)

colors = "RGB"
bayer = CFA
pattern = colors[bayer[0,0]] + colors[bayer[0,1]] + colors[bayer[1,0]] + colors[bayer[1,1]]

#%% Demosaicing using Menon 2007

# base = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(ref_img, pattern=pattern)  # pattern is [[G,R], [B,G]] for the Samsung G8
base = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(ref_img, pattern=pattern)

plt.figure("Menon 2007")
postprocessed_menon = raw2rgb.postprocess(raw_ref_img, base, do_tonemapping=True,xyz2cam=xyz2cam)
plt.imshow(postprocessed_menon, interpolation = 'none')
plt.xticks([])
plt.yticks([])

#%% Demosaicing using demosaicnet
import demosaicnet

demosaicnet_bayer = demosaicnet.BayerDemosaick()
shifted_image = ref_img[:, 1:-1]
mosaicnet_image = np.zeros((shifted_image.shape[0], shifted_image.shape[1], 3))


mosaicnet_pattern='GRBG'
mosaicnet_channel = {'R':0, 'G':1, 'B':2}
# epect GrGb
for z in range(4):
    i = z//2
    j = z%2
    channel = mosaicnet_pattern[z]
    mosaicnet_image[i::2, j::2, mosaicnet_channel[channel]] += shifted_image[i::2, j::2]
    
mosaic = mosaicnet_image.transpose((2,0,1)).astype(np.float32)

with th.no_grad():
    mosaicnet_output = demosaicnet_bayer(th.from_numpy(mosaic).unsqueeze(0)).squeeze(0).cpu().numpy()
mosaicnet_output = np.clip(mosaicnet_output, 0, 1).transpose(1,2,0).astype(np.float32)

plt.figure('demosaicnet output')
postprocessed_mosaicnet = raw2rgb.postprocess(raw_ref_img, mosaicnet_output, xyz2cam=xyz2cam, do_tonemapping=True)
# plt.imshow(postprocessed_mosaicnet, interpolation = 'none')   
# plt.xticks([])
# plt.yticks([])
    

#%% Upscale with bicubic

plt.figure('demosaicnet bicubic')
postprocessed_bicubic = cv2.resize(postprocessed_mosaicnet, None, fx = params['scale'], fy = params['scale'], interpolation=cv2.INTER_CUBIC)
plt.imshow(postprocessed_bicubic)
#%% Upscale with nearest neighboor
plt.figure('demosaicnet nearest')
postprocessed_nearest = cv2.resize(postprocessed_mosaicnet, None, fx = params["scale"], fy = params["scale"], interpolation=cv2.INTER_NEAREST)
plt.imshow(postprocessed_nearest)

#%% Kernel elements visualisation

img_grey = compute_grey_images(raw_ref_img.raw_image.copy()/1023, method="decimating")


iso = int(str(tags['Image ISOSpeedRatings']))
iso /= 100
iso = max(1, iso)

alpha = sum([x[0] for x in tags['Image Tag 0xC761'].values[::2]])/3
beta = sum([x[0] for x in tags['Image Tag 0xC761'].values[1::2]])/3
# alpha = 1.80710882e-4
# beta = 3.1937599182128e-6


img_grey = img_grey.copy_to_host()

img_grey = 2/alpha * iso**2 * np.sqrt(alpha*img_grey/iso + 3/8 * alpha**2 + beta)

img_grey = cuda.to_device(img_grey)



DEFAULT_CUDA_FLOAT_TYPE = float32
DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_NUMPY_FLOAT_TYPE = np.float32
params = get_params(30)

k_detail = params['merging']['tuning']['k_detail']
k_denoise = params['merging']['tuning']['k_denoise']
D_th = params['merging']['tuning']['D_th']
D_tr = params['merging']['tuning']['D_tr']
k_stretch = params['merging']['tuning']['k_stretch']
k_shrink = params['merging']['tuning']['k_shrink']


th_grey_img = th.as_tensor(img_grey, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
grey_imshape = grey_imshape_y, grey_imshape_x = th_grey_img.shape[2:]


grad_kernel1 = np.array([[[[-0.5, 0.5]]],
                          
                          [[[ 0.5, 0.5]]]])
grad_kernel1 = th.as_tensor(grad_kernel1, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")

grad_kernel2 = np.array([[[[0.5], 
                            [0.5]]],
                          
                          [[[-0.5], 
                            [0.5]]]])
grad_kernel2 = th.as_tensor(grad_kernel2, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")


tmp = F.conv2d(th_grey_img, grad_kernel1)
th_full_grad = F.conv2d(tmp, grad_kernel2, groups=2)
# The default padding mode reduces the shape of grey_img of 1 pixel in each
# direction, as expected

cuda_full_grads = cuda.as_cuda_array(th_full_grad.squeeze().transpose(0,1).transpose(1, 2))

l = cuda.device_array(grey_imshape + (2,), DEFAULT_NUMPY_FLOAT_TYPE)
A = cuda.device_array(grey_imshape, DEFAULT_NUMPY_FLOAT_TYPE)
D = cuda.device_array(grey_imshape, DEFAULT_NUMPY_FLOAT_TYPE)

@cuda.jit
def cuda_estimate_kernel_elements(full_grads, l, A, D,
                                  k_detail, k_denoise, D_th, D_tr):
    pixel_idx, pixel_idy = cuda.grid(2)
    
    imshape_y, imshape_x, _ = l.shape
    

    
    if (0 <= pixel_idy < imshape_y and 0 <= pixel_idx < imshape_x) :
        structure_tensor = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
        structure_tensor[0, 0] = 0
        structure_tensor[0, 1] = 0
        structure_tensor[1, 0] = 0
        structure_tensor[1, 1] = 0
        
    
        for i in range(0, 2):
            for j in range(0, 2):
                x = pixel_idx - 1 + j
                y = pixel_idy - 1 + i
                
                if (0 <= y < full_grads.shape[0] and
                    0 <= x < full_grads.shape[1]):
                    
                    full_grad_x = full_grads[y, x, 0]
                    full_grad_y = full_grads[y, x, 1]
    
                    structure_tensor[0, 0] += full_grad_x * full_grad_x
                    structure_tensor[1, 0] += full_grad_x * full_grad_y
                    structure_tensor[0, 1] += full_grad_x * full_grad_y
                    structure_tensor[1, 1] += full_grad_y * full_grad_y
        
        local_l = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e1 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        e2 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

        get_eighen_elmts_2x2(structure_tensor, local_l, e1, e2)
        
        l1 = local_l[0]
        l2 = local_l[1]
        
        l[pixel_idy, pixel_idx, 0] = l1
        l[pixel_idy, pixel_idx, 1] = l2
        
        
        
        A[pixel_idy, pixel_idx] = 1 + sqrt((l1 - l2)/(l1 + l2))

        D[pixel_idy, pixel_idx] = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1)
        
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(grey_imshape_x/threadsperblock[1]))
blockspergrid_y = int(np.ceil(grey_imshape_y/threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

cuda_estimate_kernel_elements[blockspergrid, threadsperblock](cuda_full_grads, l, A, D,
                              k_detail, k_denoise, D_th, D_tr)

#%% Plot kernel elements

l1 = l[:,:,0].copy_to_host()
plt.figure('l1')
plt.title('l1')
plt.imshow(l1, vmin= 0.5, vmax=3)
plt.colorbar()

plt.figure('A')
plt.imshow(A)
plt.title('A')
plt.colorbar()

plt.figure('D')
plt.imshow(D)
plt.title('D')
plt.colorbar()

plt.figure('gradx')
plt.title('gradx')
plt.imshow(cuda_full_grads[:,:,0])
plt.colorbar() 

plt.figure('grady')
plt.title('grady')
plt.imshow(cuda_full_grads[:,:,1])
plt.colorbar()     


#%% plot flow
maxrad = 10
flow = debug_dict['flow']

for image_index in range(comp_images.shape[0]):
    flow_image = flow2img(flow[image_index], maxrad)
    plt.figure("flow {}".format(image_index))
    plt.imshow(flow_image)

# Making color wheel
X = np.linspace(-maxrad, maxrad, 1000)
Y = np.linspace(-maxrad, maxrad, 1000)
Xm, Ym = np.meshgrid(X, Y)
Z = np.stack((Xm, Ym)).transpose(1,2,0)

flow_image = flow2img(Z, maxrad)
plt.figure("wheel")
plt.imshow(flow_image)


#%% Visualizing robustness

plt.figure('accumulated r')
r_acc = debug_dict['accumulated robustness']
plt.imshow(r_acc, cmap = "gray", interpolation='none')
clb=plt.colorbar()
clb.ax.tick_params() 
clb.ax.set_title('Accumulated\nrobustness',fontsize=15)
plt.xticks([])
plt.yticks([])
