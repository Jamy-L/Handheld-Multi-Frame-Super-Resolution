# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_rgb

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda
from time import time
import cupy as cp
from scipy.interpolate import interp2d


def merge(ref_img, comp_imgs, alignments, options, params):
    """
    Merges all the images, based on the alignments previously estimated.
    The size of the merge_result is adjustable with params['scale']


    Parameters
    ----------
    ref_img : Array[imsize_y, imsize_x]
        The reference image
    comp_imgs : Array [n_images,imsize_y, imsize_x]
        The compared images
    alignments : Array[n_images, n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment
    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    merge_result : Array[scale * imsize_y, scale * imsize_x, 3]
        merged images

    """
    # TODO Non int scale is broken atm
    VERBOSE = options['verbose']
    SCALE = params['scale']
    TILE_SIZE = params['tuning']['tileSizes']
    N_IMAGES, N_TILES_Y, N_TILES_X, _ \
        = alignments.shape

    if VERBOSE > 2:
        print('Beginning mergin process')
        current_time = time()

    native_im_size = ref_img.shape
    output_size = (SCALE*native_im_size[0], SCALE*native_im_size[1])
    output_img = cuda.device_array(output_size)

    # moving arrays to GPU
    cuda_comp_imgs = cuda.to_device(comp_imgs)
    cuda_ref_img = cuda.to_device(ref_img)
    cuda_alignments = cuda.to_device(alignments)

    # specifying the block size
    # 1 block per output pixel, 9 threads per block
    threadsperblock = (3, 3)
    blockspergrid = output_size

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data transfered to GPU')

    @cuda.jit(device=True)
    def extract_neighborhood(subtile_center_idy, subtile_center_idx, tile, neighborhood):
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (0 <= subtile_center_idy + i < TILE_SIZE and
                        0 <= subtile_center_idx + j < TILE_SIZE):

                    neighborhood[i + 2, j + 2] = tile[subtile_center_idy + i,
                                                      subtile_center_idx + j]
                else:
                    # TODO Nan would be better but I don't know how to create Nan with cuda
                    neighborhood[i + 2, j + 2] = np.inf
                    
    @cuda.jit(device=True)
    def get_closest_flows():
        #TODO 
        return None     
    
    @cuda.jit(device=True)
    def get_channel():
        #TODO 
        return None
    
    @cuda.jit(device=True)
    def ker(d, cov):
        #TODO 
        return 1
    
      
    @cuda.jit
    def accumulate(ref_img, comp_imgs, alignments, output_img):
        """
        Cuda kernel, each block represents an output pixel. Each block contains
        a 3 by 3 neighborhood for each moving image. A single threads takes
        care of one of these pixels, for all the moving images.



        Parameters
        ----------
        ref_img : Array[imsize_y, imsize_x]
            The reference image
        comp_imgs : Array[n_images-1, imsize_y, imsize_x]
            The compared images
        alignements : Array[n_images, n_tiles_y, n_tiles_x, 2]
            The alignemtn vectors for each tile of each image
        output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
            The empty output image

        Returns
        -------
        None.

        """

        output_pixel_idx, output_pixel_idy = cuda.blockIdx.x, cuda.blockIdx.y
        tx = cuda.threadIdx.x-1
        ty = cuda.threadIdx.y-1
        output_size_y, output_size_x = output_img.shape
        input_size_y, input_size_x = ref_img.shape
        
        # We pick one single thread to do certain calculations
        if tx == 0 and ty == 0:
            coarse_ref_sub_x = cuda.sharred.array(output_pixel_idx / SCALE, type=float32)
            coarse_ref_sub_y = cuda.sharred.array(output_pixel_idy / SCALE, type=float32)
            local_optical_flows = get_closest_flows(coarse_ref_sub_x,
                                                    coarse_ref_sub_y,
                                                    alignments
                                                    )
            acc = cuda.sharred.array(3, type=float64)
            acc[0] = 0
            acc[1] = 0
            acc[2] = 2

            val = cuda.sharred.array(3, type=float64)
            val[0] = 0
            val[1] = 0
            val[2] = 2
        # We need to wait the fetching of the flow
        cuda.syncthreads()

        patch_center_x = cuda.sharred.array(1, uint16)
        patch_center_y = cuda.sharred.array(1, uint16)

        for image_index in range(N_IMAGES + 1):
            if tx == 0 and ty == 0:
                if image_index == 0:  # ref image
                    # no optical flow
                    patch_center_x = round(coarse_ref_sub_x)
                    patch_center_y = round(coarse_ref_sub_y)

                else:
                    patch_center_x = round(coarse_ref_sub_x + local_optical_flows[image_index - 1, 0])
                    patch_center_y = round(coarse_ref_sub_y + local_optical_flows[image_index - 1, 1])

                # TODO compute R and kernel
                cov = None
                R = 1

            # We need to wait the calculation of R, kernel and new position
            cuda.syncthreads()
            
            patch_pixel_idx = patch_center_x + tx
            patch_pixel_idy = patch_center_y + ty
            
            if not(0 <= patch_pixel_idx < input_size_x and
                   0 <= patch_pixel_idy < input_size_y):
                return
            if image_index == 0:
                c = ref_img[patch_pixel_idy, patch_pixel_idx]
            else:
                c = comp_imgs[image_index - 1, patch_pixel_idy, patch_pixel_idx]

            channel = get_channel(patch_pixel_idx, patch_pixel_idy)

            fine_sub_pos_x = SCALE * (patch_pixel_idx - local_optical_flows[image_index - 1, 0])
            fine_sub_pos_y = SCALE * (patch_pixel_idy - local_optical_flows[image_index - 1, 1])

            dist = np.sqrt(
                (fine_sub_pos_x - output_pixel_idx) * (fine_sub_pos_x - output_pixel_idx) -
                (fine_sub_pos_y - output_pixel_idy) * (fine_sub_pos_y - output_pixel_idy))

            w = ker(dist, cov)
            
            cuda.atomic.add(val, channel, w*c*R)
            cuda.atomic.add(acc, channel, w*R)
            
        # We need to wait that every 9 pixel from every image
        # has accumulated
        cuda.syncthreads()
        
        if tx == 0 and ty == 0:
            output_img[output_pixel_idy, output_pixel_idx, 0] = val[0]/acc[0]
            output_img[output_pixel_idy, output_pixel_idx, 1] = val[1]/acc[1]
            output_img[output_pixel_idy, output_pixel_idx, 2] = val[2]/acc[2]



    accumulate[blockspergrid, threadsperblock](
        cuda_ref_img, cuda_comp_imgs, cuda_alignments, output_img)

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data merged on GPU side')

    # TODO This can probably be modified to save memory, but so far it's good
    # for debuging
    merge_result = output_img.copy_to_host()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data returned from GPU')

    # TODO 1023 is the max value in the example. Maybe we have to extract
    # metadata to have a more general framework
    # return merge_result/(1023*accumulator)
    return merge_result

# %% test code, to remove in the final version


ref_img = rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/33TJ_20150606_224837_294/payload_N000.dng'
                       ).raw_image.copy()


comp_images = rawpy.imread(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N001.dng').raw_image.copy()[None]
for i in range(2, 9):
    comp_images = np.append(comp_images, rawpy.imread('C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/test_data/33TJ_20150606_224837_294/payload_N00{}.dng'.format(i)
                                                      ).raw_image.copy()[None], axis=0)

pre_alignment = np.load(
    'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus_python/results_test1/unpaddedMotionVectors.npy')[:-1]
n_images, n_patch_y, n_patch_x, _ = pre_alignment.shape
tile_size = 32
native_im_size = ref_img.shape

params = {'tuning': {'tileSizes': 32}, 'scale': 2}
params['tuning']['kanadeIter'] = 3


t1 = time()
final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_images, pre_alignment, {"verbose": 3}, params)

comp_tiles = np.zeros((n_images, n_patch_y, n_patch_x, tile_size, tile_size))
for i in range(n_images):
    comp_tiles[i] = getAlignedTiles(
        comp_images[i], tile_size, final_alignments[i])


output = merge(ref_img, comp_tiles, final_alignments, {"verbose": 3}, params)
print('\nTotal ellapsed time : ', time() - t1)
