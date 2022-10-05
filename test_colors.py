# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 18:05:05 2022

@author: jamyl
"""
import numpy as np
import exifread
import rawpy
import matplotlib.pyplot as plt
from handheld_super_resolution.raw2rgb import process_isp

im_path = 'P:/0001/Samsung/im_00.dng'
raw = rawpy.imread(im_path)
raw_im = rawpy.imread(im_path).raw_image.copy()/1023

with open(im_path, 'rb') as raw_file:
    tags = exifread.process_file(raw_file)
    
CFA = str((tags['Image CFAPattern']))[1:-1].split(sep=', ')
CFA = np.array([int(x) for x in CFA]).reshape(2,2)

rgb_im = np.empty((int(raw_im.shape[0]/2), int(raw_im.shape[1]/2), 3))
rgb_im[:,:,0] = raw_im[::2, 1::2]
rgb_im[:,:,1] = (raw_im[::2, ::2] + raw_im[1::2, 1::2])/2
rgb_im[:,:,2] = raw_im[1::2, ::2]



colormatrix1 = np.empty((3,3))
colormatrix2 = np.empty_like(colormatrix1)
forwardmatrix1 = np.empty((3,3))
forwardmatrix2 = np.empty_like(colormatrix1)

for k in range(9):
    i = k//3
    j = k%3
    forwardmatrix1[i,j] = float(tags["Image Tag 0xC714"].values[k])
    forwardmatrix2[i,j] = float(tags["Image Tag 0xC715"].values[k])
    colormatrix1[i,j] = float(tags["Image Tag 0xC621"].values[k])
    colormatrix2[i,j] = float(tags["Image Tag 0xC622"].values[k])

b = process_isp(raw, rgb_im)
plt.figure("test")
plt.imshow(b)
plt.figure("raw")
plt.imshow(rgb_im)
plt.figure("postprocessed")
plt.imshow(raw.postprocess())




