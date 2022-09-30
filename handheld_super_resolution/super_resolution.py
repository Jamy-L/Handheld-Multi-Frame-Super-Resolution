# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:56:22 2022

@author: jamyl
"""

import os
import glob
from time import time

import numpy as np
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
import exifread
import rawpy
import matplotlib.pyplot as plt

from handheld_super_resolution.merge import merge
from handheld_super_resolution.block_matching import alignHdrplus
from handheld_super_resolution.optical_flow import lucas_kanade_optical_flow_V2
from handheld_super_resolution.utils import getTime
from handheld_super_resolution.robustness import compute_robustness
from handheld_super_resolution import DEFAULT_NUMPY_FLOAT_TYPE


def main(ref_img, comp_imgs, options, params):
    verbose = options['verbose'] > 1
    verbose_2 = options['verbose'] >2
    
    
    t1 = time()
    if verbose :
        print('Beginning block matching')
    pre_alignment, aligned_tiles = alignHdrplus(ref_img, comp_imgs,params['block matching'], options)
    pre_alignment = pre_alignment[:, :, :, ::-1] # swapping x and y direction (x must be first)
    if verbose : 
        current_time = getTime(t1, 'Block Matching')
    
    
    cuda_ref_img = cuda.to_device(ref_img)
    cuda_comp_imgs = cuda.to_device(comp_imgs)
    
    if verbose_2 : 
        current_time = getTime(
            current_time, 'Arrays moved to GPU')
    
    
    cuda_final_alignment = lucas_kanade_optical_flow_V2(
        ref_img, comp_imgs, pre_alignment, options, params['kanade'])
    
    
    if verbose : 
        current_time = time()
        
    cuda_Robustness, cuda_robustness = compute_robustness(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment,
                                             options, params['merging'])
    if verbose : 
        current_time = getTime(
            current_time, 'Robustness estimated')
    
    output = merge(cuda_ref_img, cuda_comp_imgs, cuda_final_alignment, cuda_robustness, {"verbose": 3}, params['merging'])
    print('\nTotal ellapsed time : ', time() - t1)
    return output, cuda_Robustness.copy_to_host(), cuda_robustness.copy_to_host()
    
#%%

def process(burst_path, options, params):
    currentTime, verbose = time(), options['verbose'] > 2
    

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
    raw_comp = np.array(raw_comp)
    # Reference image selection and metadata         
    ref_raw = rawpy.imread(raw_path_list[ref_id]).raw_image.copy()
    with open(raw_path_list[ref_id], 'rb') as raw_file:
        tags = exifread.process_file(raw_file)
    
    if not 'exif' in params['merging'].keys(): 
        params['merging']['exif'] = {}
        
    params['merging']['exif']['white level'] = int(str(tags['Image Tag 0xC61D']))
    CFA = str((tags['Image CFAPattern']))[1:-1].split(sep=', ')
    CFA = np.array([int(x) for x in CFA]).reshape(2,2)
    params['merging']['exif']['CFA Pattern'] = CFA
    
    if verbose:
        currentTime = getTime(currentTime, ' -- Read raw files')
    
    # casting type fom int to float if necessary. Normalisation into [0, 1]
    if np.issubdtype(type(ref_raw[0,0]), np.integer):
        ref_raw = (ref_raw/params['merging']['exif']['white level']).astype(DEFAULT_NUMPY_FLOAT_TYPE)
    if np.issubdtype(type(raw_comp[0,0,0]), np.integer):
        raw_comp = (raw_comp/params['merging']['exif']['white level']).astype(DEFAULT_NUMPY_FLOAT_TYPE)
        
    return main(ref_raw, raw_comp, options, params)