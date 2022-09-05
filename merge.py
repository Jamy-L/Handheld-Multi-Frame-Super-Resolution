# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:07 2022

@author: jamyl
"""

from optical_flow import lucas_kanade_optical_flow
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
from hdrplus_python.package.algorithm.merging import depatchifyOverlap
from hdrplus_python.package.algorithm.genericUtils import getTime
from kernels import compute_kernel_cov

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
    output_img = cuda.device_array(output_size+(3,)) #third dim for rgb canals

    # moving arrays to GPU
    cuda_comp_imgs = cuda.to_device(comp_imgs)
    cuda_ref_img = cuda.to_device(ref_img)
    cuda_alignments = cuda.to_device(alignments)

    # specifying the block size
    # 1 block per output pixel, 9 threads per block
    threadsperblock = (3, 3)
    # we need to swap the shape to have idx horiztonal
    blockspergrid = (output_size[1], output_size[0])

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
    def get_closest_flow(idx_sub, idy_sub, optical_flows, local_flow):
        """
        Returns the estimated optical flow for a subpixel (or a pixel), based on
        the tile based estimated optical flow.

        Parameters
        ----------
        idx_sub, idy_sub : float
            subpixel ids where the flow must be estimated
        optical_flow : Array[n_tiles_y, n_tiles_x, 2]
            tile based optical flow


        Returns
        -------
        flow : array[2]
            optical flow

        """
        patch_idy_bottom = int(idy_sub//(TILE_SIZE//2))
        patch_idy_top = patch_idy_bottom - 1

        patch_idx_right = int(idx_sub//(TILE_SIZE//2))
        patch_idx_left = patch_idx_right - 1

        imshape = optical_flows.shape[:2]
        # out of bounds. With zero flow, they will be discarded later
        if (idx_sub < 0 or idx_sub >= native_im_size[1] or
            idy_sub < 0 or idy_sub >= native_im_size[0]):
            flow_x = 0
            flow_y = 0

        # corner conditions
        elif patch_idy_bottom >= imshape[0] and patch_idx_left < 0:
            flow_x = optical_flows[patch_idy_top, patch_idx_right, 0]
            flow_y = optical_flows[patch_idy_top, patch_idx_right, 1]
    
        elif patch_idy_bottom >= imshape[0] and patch_idx_right >= imshape[1]:
            flow_x = optical_flows[patch_idy_top, patch_idx_left, 0]
            flow_y = optical_flows[patch_idy_top, patch_idx_left, 1]
    
        elif patch_idy_top < 0 and patch_idx_left < 0:
            flow_x = optical_flows[patch_idy_bottom, patch_idx_right, 0]
            flow_y = optical_flows[patch_idy_bottom, patch_idx_right, 1]
    
        elif patch_idy_top < 0 and patch_idx_right >= imshape[1]:
            flow_x = optical_flows[patch_idy_bottom, patch_idx_left, 0]
            flow_y = optical_flows[patch_idy_bottom, patch_idx_left, 1]
    
        # side conditions
        elif patch_idy_bottom >= imshape[0]:
            flow_x = (optical_flows[patch_idy_top, patch_idx_left, 0] +
                    optical_flows[patch_idy_top, patch_idx_right, 0])/2
            flow_y = (optical_flows[patch_idy_top, patch_idx_left, 1] +
                    optical_flows[patch_idy_top, patch_idx_right, 1])/2
    
        elif patch_idy_top < 0:
            flow_x = (optical_flows[patch_idy_bottom, patch_idx_left, 0] +
                    optical_flows[patch_idy_bottom, patch_idx_right, 0])/2
            flow_y = (optical_flows[patch_idy_bottom, patch_idx_left, 1] +
                    optical_flows[patch_idy_bottom, patch_idx_right, 1])/2
    
        elif patch_idx_left < 0:
            flow_x = (optical_flows[patch_idy_bottom, patch_idx_right, 0] +
                    optical_flows[patch_idy_top, patch_idx_right, 0])/2
            flow_y = (optical_flows[patch_idy_bottom, patch_idx_right, 1] +
                    optical_flows[patch_idy_top, patch_idx_right, 1])/2
    
        elif patch_idx_right >= imshape[1]:
            flow_x = (optical_flows[patch_idy_bottom, patch_idx_left, 0] +
                    optical_flows[patch_idy_top, patch_idx_left, 0])/2
            flow_y = (optical_flows[patch_idy_bottom, patch_idx_left, 1] +
                    optical_flows[patch_idy_top, patch_idx_left, 1])/2
    
        # general case
        else:
            # Averaging patches
            flow_x = (optical_flows[patch_idy_top, patch_idx_left, 0] +
                    optical_flows[patch_idy_top, patch_idx_right, 0] +
                    optical_flows[patch_idy_bottom, patch_idx_left, 0] +
                    optical_flows[patch_idy_bottom, patch_idx_right, 0])/4
            flow_y = (optical_flows[patch_idy_top, patch_idx_left, 1] +
                    optical_flows[patch_idy_top, patch_idx_right, 1] +
                    optical_flows[patch_idy_bottom, patch_idx_left, 1] +
                    optical_flows[patch_idy_bottom, patch_idx_right, 1])/4
        
     
        local_flow[0] = flow_x
        local_flow[1] = flow_y
    
    @cuda.jit(device=True)
    def get_channel(patch_pixel_idx, patch_pixel_idy):
        """
        Return 0, 1 or 2 depending if the coordinates point a red, green or
        blue pixel on the Bayer frame

        Parameters
        ----------
        patch_pixel_idx : unsigned int
            horizontal coordinates
        patch_pixel_idy : unigned int
            vertical coordinates

        Returns
        -------
        int

        """
        if patch_pixel_idx%2 == 1 and patch_pixel_idy%2 == 1: #R
            return 0
        
        elif patch_pixel_idx%2 == 1 and patch_pixel_idy%2 == 0: #G
            return 1

        elif patch_pixel_idx%2 == 0 and patch_pixel_idy%2 == 1: #G
            return 1
        
        elif patch_pixel_idx%2 == 0 and patch_pixel_idy%2 ==0 :#B
            return 2
        
    
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
        output_size_y, output_size_x, _ = output_img.shape
        input_size_y, input_size_x = ref_img.shape
        
        acc = cuda.shared.array(3, dtype=float64)
        val = cuda.shared.array(3, dtype=float64)
        
        # We pick one single thread to do certain operations
        if tx == 0 and ty == 0:
            coarse_ref_sub_x = cuda.shared.array(1, dtype=float64)
            coarse_ref_sub_x[0] = output_pixel_idx / SCALE
            
            coarse_ref_sub_y = cuda.shared.array(1, dtype=float64)
            coarse_ref_sub_y[0] = output_pixel_idy / SCALE

            
            acc[0] = 1
            acc[1] = 1
            acc[2] = 1

            val[0] = 1
            val[1] = 1
            val[2] = 1
        # We need to wait the fetching of the flow
        cuda.syncthreads()

        patch_center_x = cuda.shared.array(1, uint16)
        patch_center_y = cuda.shared.array(1, uint16)
    

        for image_index in range(N_IMAGES + 1):
            if tx == 0 and ty == 0:
                if image_index == 0:  # ref image
                    # no optical flow
                    patch_center_x[0] = uint16(round(coarse_ref_sub_x[0]))
                    patch_center_y[0] = uint16(round(coarse_ref_sub_y[0]))

                else:
                    local_optical_flow = cuda.shared.array(2, dtype=float64)
                    get_closest_flow(coarse_ref_sub_x[0],
                                      coarse_ref_sub_y[0],
                                      alignments[image_index - 1],
                                      local_optical_flow)
                    
                    
                    patch_center_x[0] = uint16(round(coarse_ref_sub_x[0] + local_optical_flow[0]))
                    patch_center_y[0] = uint16(round(coarse_ref_sub_y[0] + local_optical_flow[1]))
            
            # we need the position of the patch before computing the kernel
            cuda.syncthreads()
            
            # TODO compute R and kernel with another thread
            cov = cuda.syncthreads((2, 2), dtype=float64)
            if image_index == 0:
                compute_kernel_cov(ref_img, patch_center_x[0], patch_center_y[0], cov)
            else:
                compute_kernel_cov(comp_imgs[image_index - 1], patch_center_x[0], patch_center_y[0], cov)
            R = 1

            # We need to wait the calculation of R and kernel
            cuda.syncthreads()
            
            patch_pixel_idx = patch_center_x[0] + tx
            patch_pixel_idy = patch_center_y[0] + ty
            
            # in bounds conditions
            if (0 <= patch_pixel_idx < input_size_x and
                0 <= patch_pixel_idy < input_size_y):
                if image_index == 0:
                    c = ref_img[patch_pixel_idy, patch_pixel_idx]
                else:
                    c = comp_imgs[image_index - 1, patch_pixel_idy, patch_pixel_idx]
                    
                # checking if pixel is r, g or b
                channel = get_channel(patch_pixel_idx, patch_pixel_idy)
    
                # applying invert transformation and upscaling
                fine_sub_pos_x = SCALE * (patch_pixel_idx - local_optical_flow[0])
                fine_sub_pos_y = SCALE * (patch_pixel_idy - local_optical_flow[1])
                

                dist = math.sqrt(
                    (fine_sub_pos_x - output_pixel_idx) * (fine_sub_pos_x - output_pixel_idx) +
                    (fine_sub_pos_y - output_pixel_idy) * (fine_sub_pos_y - output_pixel_idy))
    
                w = ker(dist, cov)


                cuda.atomic.add(val, channel, c*w*R)
                cuda.atomic.add(acc, channel, w*R)
            
            # We need to wait that every 9 pixel from every image
            # has accumulated
            cuda.syncthreads()
        if tx == 0 and ty == 0:
            output_img[output_pixel_idy, output_pixel_idx, 0] = val[0]/acc[0]
            output_img[output_pixel_idy, output_pixel_idx, 1] = val[1]/acc[1]
            output_img[output_pixel_idy, output_pixel_idx, 2] = val[2]/acc[2]
 
######

    accumulate[blockspergrid, threadsperblock](
        cuda_ref_img, cuda_comp_imgs, cuda_alignments, output_img)

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data merged on GPU side')

    
    merge_result = output_img.copy_to_host()

    if VERBOSE > 2:
        current_time = getTime(
            current_time, ' \t- Data returned from GPU')

    # TODO 1023 is the max value in the example. Maybe we have to extract
    # metadata to have a more general framework
    # return merge_result/(1023*accumulator)
    return merge_result

def gamma(image):
    return image**(1/2.2)

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

params = {'tuning': {'tileSizes': 32}, 'scale': 6}
params['tuning']['kanadeIter'] = 3


t1 = time()
final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_images, pre_alignment, {"verbose": 3}, params)

# comp_tiles = np.zeros((n_images, n_patch_y, n_patch_x, tile_size, tile_size))
# for i in range(n_images):
#     comp_tiles[i] = getAlignedTiles(
#         comp_images[i], tile_size, final_alignments[i])


output = merge(ref_img, comp_images, final_alignments, {"verbose": 3}, params)
print('\nTotal ellapsed time : ', time() - t1)

plt.imshow(gamma(output/1023))
