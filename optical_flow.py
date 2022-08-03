# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:36 2022

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


def lucas_kanade_optical_flow(ref_img, pre_aligned_tiles, options):
    """
    Computes the displacement based on a naive implementation of
    Lucas-Kanade optical flow (https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf)
    (The system solving method, not the iterative)

    Parameters
    ----------
    ref_img : Array[imsize_y,imsize_x]
        The reference image
    pre_aligned_tiles : Array[n_images, n_tiles_y,
        n_tiles_x, tile_size, tile_size]
        The rearrangement of tile obtained with after using the coarse to fine pyramidal method
    options : Dict
        Options to pass

    Returns
    -------
    displacement : Array[n_images, n_tiles_y, n_tiles_x, 2]
        displacement vector for each tile of each image

    """
    print("\t- estimating Lucas-Kanade's optical flow")
    # Estimating gradients with cv2 sobel filters
    gradx = Sobel(ref_img, 0, dx=1, dy=0)
    grady = Sobel(ref_img, 0, dx=0, dy=1)

    # hdr_plus returns in each axes 3 tiles supplementary. I have figured out
    # that the first and 2 last one should be remove, so the dimensions can be
    # coherent with the base image
    n_images, n_patch_y, n_patch_x, tile_size, _ \
        = pre_aligned_tiles[:, 1:-2, 1:-2].shape

    # dividing into tiles, solving systems will be easier to implement
    # this way
    gradx = getTiles(gradx, tile_size, tile_size // 2)
    grady = getTiles(grady, tile_size, tile_size // 2)
    ref_img_tiles = getTiles(ref_img, tile_size, tile_size // 2)

    # dummy tensor for passing shapes to numba
    dummy = np.empty((tile_size**2, 3))

    t1 = time()
    # syst[:, 0:-1] = A ; syst[:,-2] = B
    syst = _compute_tilewise_systems(
        np.ascontiguousarray(ref_img_tiles),
        np.ascontiguousarray(gradx), np.ascontiguousarray(grady),
        np.ascontiguousarray(pre_aligned_tiles[:, 1:-2, 1:-2]),
        np.ascontiguousarray(dummy))

    print('\t\t- Finished computing system arrays ', time()-t1)
    t1 = time()
    A = np.array(syst[:, :, :, :, :-1])
    B = np.array(syst[:, :, :, :, -1][None])

    B = np.transpose(B, (1, 2, 3, 4, 0))

    ATA = np.matmul(np.transpose(
        A, (0, 1, 2, 4, 3)), A)
    ATB = np.matmul(np.transpose(
        A, (0, 1, 2, 4, 3)), B)
    print('\t\t- Finished multiplying arrays ', time()-t1)
    t1 = time()
    displacement = np.linalg.solve(ATA, ATB)
    print("\t\t- system solved ", time() - t1)
    return displacement


@ guvectorize(['(uint16[:,:,:,:], uint8[:,:,:,:], uint8[:,:,:,:], uint16[:,:,:,:,:], float64[:,:], uint16[:,:,:,:,:])'],
              '(n_tiles_y, n_tiles_x, tile_size, tile_size), (n_tiles_y, n_tiles_x, tile_size, tile_size), (n_tiles_y, n_tiles_x, tile_size, tile_size), (n_images, n_tiles_y, n_tiles_x, tile_size, tile_size), (syst_size, _) -> (n_images, n_tiles_y, n_tiles_x, syst_size, _)',
              nopython=True)  # target='cuda')
def _compute_tilewise_systems(ref_img_tiles, gradx, grady, pre_aligned_tiles, dummy, syst):
    n_images, n_patch_y, n_patch_x, patch_size, _\
        = pre_aligned_tiles.shape

    for image_index in range(n_images):
        for patch_idx in range(n_patch_x):
            for patch_idy in range(n_patch_y):
                for pixel_idx in range(patch_size):
                    for pixel_idy in range(patch_size):
                        syst[image_index, patch_idy, patch_idx, pixel_idy + patch_size*pixel_idx,
                             0] = gradx[patch_idy, patch_idx, pixel_idy, pixel_idx]
                        syst[image_index, patch_idy, patch_idx, pixel_idy + patch_size*pixel_idx,
                             1] = grady[patch_idy, patch_idx, pixel_idy, pixel_idx]
                        syst[image_index, patch_idy, patch_idx, pixel_idy + patch_size*pixel_idx,
                             2] = ref_img_tiles[patch_idy, patch_idx, pixel_idy, pixel_idx] \
                            - pre_aligned_tiles[image_index, patch_idy,
                                                patch_idx, pixel_idy, pixel_idx]


ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/payload_N000.dng'
                       ).raw_image.copy()


pre_aligned_tiles = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/33TJ_20150606_224837_294_aligned_tiles.npy')


c = lucas_kanade_optical_flow(ref_img, pre_aligned_tiles, None)

# %%
n_images, n_patch_y, n_patch_x, tile_size, _ \
    = pre_aligned_tiles[:, 1:-2, 1:-2].shape
native_im_size = ref_img.shape
# ref_im_r = ref_im
scale = 3

final_frame = np.zeros(
    (3, scale*native_im_size[0], scale*native_im_size[1]))  # * float('Nan')

final_frame[0, scale::scale*2, scale::scale*2] = ref_img[1::2, 1::2]  # r
final_frame[1, scale::scale*2, ::scale*2] = ref_img[1::2, 0::2]  # g1
final_frame[1, 0::scale*2, scale::scale*2] = ref_img[0::2, 1::2]  # g2
final_frame[2, 0::scale*2, 0::scale*2] = ref_img[0::2, 0::2]  # b

# %%
# upscaling displacement and int convertion
disp = np.round(c*scale)

count = 0
for im_id in tqdm(range(1, 10)):
    for tile_idy in range(n_patch_y):
        for tile_idx in range(n_patch_x):
            for channel in range(3):
                tile = pre_aligned_tiles[im_id, tile_idy, tile_idx]

                start_idy = int(tile_size*tile_idy//2*scale +
                                disp[im_id, tile_idy, tile_idx, 1])
                end_idy = start_idy + scale*tile_size
                start_idx = int(tile_size*tile_idx//2*scale +
                                disp[im_id, tile_idy, tile_idx, 1])
                end_idx = start_idx + scale*tile_size
                if start_idx >= 0 and end_idx < native_im_size[1]*scale and start_idy >= 0 and end_idy < native_im_size[0]*scale:
                    final_frame[0, (start_idy+scale):end_idy:scale*2,
                                (start_idx+scale):end_idx:scale*2] = tile[1::2, 1::2]  # r

                    final_frame[1, start_idy:end_idy:scale*2, (start_idx +
                                scale):end_idx:scale*2] = tile[0::2, 1::2]  # g1

                    final_frame[1, (start_idy+scale):end_idy:scale*2,
                                start_idx:end_idx:scale*2] = tile[1::2, 0::2]  # g2

                    final_frame[2, start_idy:end_idy:scale*2,
                                start_idx:end_idx:scale*2] = tile[1::2, 1::2]  # b
                    # print('pouet')
                else:
                    # print(count)
                    count += 1
