# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap

from cv2 import Sobel
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit
from time import time


def merge(ref_img, pre_kanade_tiles, alignment, factor, options):
    final_shape = (ref_img.shape[0]*factor, ref_img.shape[1]*factor)
    final_image = np.empty(final_shape).float('Nan')
    accumulator = np.empty_like(final_image)

    n_images, n_tiles_y, n_tiles_x, tile_size, _ = pre_kanade_tiles.shape
    for img_indes in range(n_images):
        for tile_idy in range(n_tiles_y):
            for tile_idx in range(n_tiles_x):
                # 1 calculating the upscale coordinates, considering the alignment

                # 2

    return final_image


def plot_merge(final_image):
    final_image[np.isnan(final_image)] = 0
    plt.imshow(final_image)


# %%
ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/payload_N000.dng'
                       ).raw_image.copy()
pre_kanade_tiles = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/33TJ_20150606_224837_294_aligned_tiles.npy')

alignment = ''
