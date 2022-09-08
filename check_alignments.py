# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:27:27 2022

@author: jamyl
"""

from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
import rawpy
import numpy as np
from optical_flow import lucas_kanade_optical_flow
import matplotlib.pyplot as plt

ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/payload_N000.dng'
                       ).raw_image.copy()


comp_images = rawpy.imread(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
for i in range(1, 10):
    comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)

pre_alignment = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')
n_images, n_patch_y, n_patch_x, _ = pre_alignment.shape
tile_size = 32
native_im_size = ref_img.shape

params = {'tuning': {'tileSizes': 32}, 'scale': 2}
params['tuning']['kanadeIter'] = 3


final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_images, pre_alignment, {"verbose": 3}, params)
# %%
comp_tiles = []
for i in range(0, 10):
    comp_tiles.append(
        getAlignedTiles(
            comp_images[i], tile_size, final_alignments[i-1])[::2, ::2])

    test = comp_tiles[-1].transpose((0, 2, 1, 3))
    test = test.reshape(97, 32, 32*131).transpose(2, 0,
                                                  1).reshape(32*131, 97*32).transpose()
    plt.figure()
    plt.imshow(test, cmap='gray')