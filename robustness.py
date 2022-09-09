# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_kernel_cov
from linalg import quad_mat_prod

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
from time import time
from math import isnan, sqrt, exp

def compute_robustness(ref_img, comp_imgs, flows, sigma_t, dt, Mt, s1, s2, t):
    
    n_images, imshape_y, imshape_x = comp_imgs.shape
    gray_imshape_y, gray_imshape_x = int(imshape_y/2), int(imshape_x/2)
    
    r = cuda.device_array((gray_imshape_y, gray_imshape_x))
    R = cuda.device_array((gray_imshape_y, gray_imshape_x))
    
    @cuda.jit(device=True)
    def compute_guide_patchs(ref_img, comp_imgs, flows, guide_patch_ref, guide_patch_comp):
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x -1, cuda.threadIdx.y -1
        
        
        top_left_ref_y = pixel_idy*2 +2*ty
        top_left_ref_x = pixel_idx*2 +2*tx
        if (0 <= top_left_ref_y < imshape_y -1) and (0 <= top_left_ref_x < imshape_x -1): # ref inbounds
            # ref
            guide_patch_ref[ty + 1, tx + 1, 0] = ref_img[top_left_ref_y + 1 ,
                                                         top_left_ref_x + 1] #r
            guide_patch_ref[ty + 1, tx + 1, 1] = (ref_img[top_left_ref_y, top_left_ref_x + 1] +
                                                  ref_img[top_left_ref_y + 1, top_left_ref_x])/2#g
            guide_patch_ref[ty + 1, tx + 1, 2] = ref_img[top_left_ref_y,
                                                         top_left_ref_x] #b   
            
            
            # Moving. We divide flow by 2 because grey image is twice smaller
            top_left_m_x = round(top_left_ref_x + flows[image_index, top_left_ref_y,
                                                  top_left_ref_x, 0]/2)
            top_left_m_x = top_left_m_x - top_left_m_x%2
            # This transformation insures that we are indeed arriving on the right color channel (Bayer top left is generally red or blue)
            
            top_left_m_y = round(top_left_ref_y + flows[image_index, top_left_ref_y,
                                                  top_left_ref_x, 1]/2)
            top_left_m_y = top_left_m_y - top_left_m_y%2

            
            if (0 <= top_left_m_y < imshape_y) and (0 <= top_left_m_x < imshape_x):
                guide_patch_comp[ty + 1, tx + 1, 2] = comp_imgs[image_index, top_left_m_y, top_left_m_x] # b
            else:
                guide_patch_comp[ty + 1, tx + 1, 2] = 0/0 #Nan
                
            
            if (0 <= top_left_m_y + 1< imshape_y) and (0 <= top_left_m_x + 1< imshape_x):
                guide_patch_comp[ty + 1, tx + 1, 0] = comp_imgs[image_index, top_left_m_y + 1, top_left_m_x + 1] # r
            else:
                guide_patch_comp[ty + 1, tx + 1, 0] = 0/0 #Nan
                
            if (0 <= top_left_m_y < imshape_y) and (0 <= top_left_m_x + 1< imshape_x):
                g1 = comp_imgs[image_index, top_left_m_y, top_left_m_x + 1] # g1
            else:
                g1 = 0/0 #Nan
            
            if (0 <= top_left_m_y + 1< imshape_y) and (0 <= top_left_m_x < imshape_x):
                g2 = comp_imgs[image_index, top_left_m_y + 1, top_left_m_x] # g2
            else:
                g2 = 0 / 0  # Nan

            if isnan(g1) and isnan(g2):
                guide_patch_comp[ty + 1, tx + 1, 1] = 0/0
            elif isnan(g1):
                guide_patch_comp[ty + 1, tx + 1, 1] = g2
            elif isnan(g2):
                guide_patch_comp[ty + 1, tx + 1, 1] = g1
            else:
                guide_patch_comp[ty + 1, tx + 1, 1] = (g1 + g2)/2
            
            
        else: #out of bounds
            guide_patch_comp[ty + 1, tx + 1, 0] = 0/0 #Nan
            guide_patch_comp[ty + 1, tx + 1, 1] = 0/0 #Nan
            guide_patch_comp[ty + 1, tx + 1, 2] = 0/0 #Nan
            
            guide_patch_comp[ty + 1, tx + 1, 0] = 0/0 #Nan
            guide_patch_comp[ty + 1, tx + 1, 1] = 0/0 #Nan
            guide_patch_comp[ty + 1, tx + 1, 2] = 0/0 #Nan
            
        

    @cuda.jit(device=True)
    def compute_local_stats(guide_patch_ref, guide_patch_comp,
                            local_stats_ref, local_stats_comp):
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        if not(isnan(guide_patch_ref[ty, tx])):
            cuda.atomic.add(local_stats_ref, 0, guide_patch_ref[ty, tx])
            cuda.atomic.add(local_stats_ref, 1, guide_patch_ref[ty, tx]**2)
            
        if not(isnan(guide_patch_comp[ty, tx])):
            cuda.atomic.add(local_stats_comp, 0, guide_patch_comp[ty, tx])
            cuda.atomic.add(local_stats_comp, 1, guide_patch_comp[ty, tx]**2)
        
        # TODO when everybody is Nan (because patchs are out of bound) stats
        # are staying at 0. Maybe it's not that bad because we are capping with the 
        # model's values later
    
    @cuda.jit(device=True)
    def compute_m(flows, mini, maxi, M):
        tx, ty = cuda.threadIdx.x - 1, cuda.threadIdx.y - 1
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        top_left_y = pixel_idy*2 +2*ty
        top_left_x = pixel_idx*2 +2*tx
        
        if 0 <= top_left_x < imshape_x and 0 <= top_left_y < imshape_y: #inbounds
            #local max search
            cuda.atomic.max(maxi, 0, flows[image_index, top_left_y, top_left_x, 0])
            cuda.atomic.max(maxi, 1, flows[image_index, top_left_y, top_left_x, 1])
            #local min search
            cuda.atomic.min(mini, 0, flows[image_index, top_left_y, top_left_x, 0])
            cuda.atomic.min(mini, 1, flows[image_index, top_left_y, top_left_x, 1])
            
        cuda.syncthreads()    
        if tx == 0 and ty == 0:
            M[0] = maxi[0] - mini[0]
            M[1] = maxi[1] - mini[1]
    
    
    @cuda.jit
    def cuda_compute_robustness(ref_img, comp_imgs, flows, sigma_t, dt, Mt, s1, s2, t, R, r):
        guide_patch_ref = cuda.shared.array((3, 3, 3), dtype=float64)
        guide_patch_comp = cuda.shared.array((3, 3, 3), dtype=float64)
        
        image_index, pixel_idy, pixel_idx = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        
        compute_guide_patchs(ref_img, comp_imgs, flows, guide_patch_ref, guide_patch_comp)
        local_stats_ref = cuda.shared.array(2, dtype=float64) #mu, sigma
        local_stats_ref[0] = 0
        local_stats_ref[1] = 0
        local_stats_comp = cuda.shared.array(2, dtype=float64)
        local_stats_comp[0] = 0
        local_stats_comp[1] = 0
        cuda.syncthreads()
        
        maxi = cuda.shared.array(2, dtype=float64) 
        mini = cuda.shared.array(2, dtype=float64)
        M = cuda.shared.array(2, dtype=float64)
        
        compute_local_stats(guide_patch_ref, guide_patch_comp,
                            local_stats_ref, local_stats_comp)
        cuda.syncthreads()
        if tx == 0 and ty ==0:
            # normalizing
            local_stats_ref[0] /= 9
            local_stats_ref[1] = sqrt(local_stats_ref[1]/9 -  local_stats_ref[0]**2)
            
            
            local_stats_comp[0] /= 9
            local_stats_comp[1] = sqrt(local_stats_comp[1]/9 -  local_stats_comp[0]**2)
            
            dp = abs(local_stats_ref[0] - local_stats_comp[0])
            # noise correction
            sigma = max(sigma_t, local_stats_comp[1])
            d = dp*(dp**2/(dp**2 + dt**2))
            
            # init, we take one flow vector from the patch
            maxi[0] = flows[image_index, pixel_idy*2, pixel_idx*2, 0]
            maxi[1] = flows[image_index, pixel_idy*2, pixel_idx*2, 1]
            mini[0] = flows[image_index, pixel_idy*2, pixel_idx*2, 0]
            mini[1] = flows[image_index, pixel_idy*2, pixel_idx*2, 1]
            
        cuda.syncthreads()
        compute_m(flows, mini, maxi, M)
        cuda.syncthreads()
        
        if tx == 0 and ty ==0:
            if sqrt(M[0]**2 + M[1]**2) > Mt:
                R[pixel_idy, pixel_idx] = s1*exp(-d**2/sigma**2) - t
            else:
                R[pixel_idy, pixel_idx] = s2*exp(-d**2/sigma**2) - t
        
    
    cuda_compute_robustness[(n_images, gray_imshape_y, gray_imshape_x), (3, 3)
        ](ref_img, comp_imgs, flows, sigma_t, dt, Mt, s1, s2, t, R, r)
    
    return cuda.copy_to_host(R)
    
# %% test code, to remove in the final version

raw_ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N000.dng'
                       )
ref_img = raw_ref_img.raw_image.copy()


comp_images = rawpy.imread(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
for i in range(2, 10):
    comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)

pre_alignment = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')

n_images, n_patch_y, n_patch_x, _ = pre_alignment.shape
tile_size = 32
native_im_size = ref_img.shape

params = {'tuning': {'tileSizes': 32}, 'scale': 1}
params['tuning']['kanadeIter'] = 3
t = 0.12
s1 = 12
s2 = 2
Mt = 0.8

sigma_t = 2
dt = 15




t1 = time()
final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_images, pre_alignment, {"verbose": 3}, params)

output = compute_robustness(ref_img, comp_images, final_alignments, sigma_t, dt, Mt, s1, s2, t)
print('\nTotal ellapsed time : ', time() - t1)