# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:27:27 2022

@author: jamyl
"""

import matplotlib
from hdrplus_python.package.algorithm.imageUtils import getTiles, getAlignedTiles
import rawpy
import numpy as np
from optical_flow import lucas_kanade_optical_flow
import matplotlib.pyplot as plt

ref_img = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        if 15**2 <= (i-50)**2 + (j-50)**2 <= 20**2:
            ref_img[i, j] = 10

comp_image = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        if 17**2 <= (i-50)**2 + (j-50)**2 <= 22**2:
            comp_image[i, j] = 10

pre_alignment = getTiles(np.zeros(ref_img.shape), 16, steps=8)
pre_alignment = np.zeros((pre_alignment.shape[0], pre_alignment.shape[1]))
pre_alignment = np.stack((pre_alignment, pre_alignment)).transpose(1, 2, 0)

tile_size = 16
native_im_size = ref_img.shape

params = {'tuning': {'tileSizes': 16}, 'scale': 2}
params['tuning']['kanadeIter'] = 3


final_alignments = lucas_kanade_optical_flow(
    ref_img, comp_image[None], pre_alignment[None], {"verbose": 3}, params)
# %%
comp_tiles = []
for i in range(0, 1):
    comp_tiles.append(
        getAlignedTiles(
            comp_image, tile_size, final_alignments[-1])[::2, ::2])

    test = comp_tiles[-1].transpose((0, 2, 1, 3))
    test = test.reshape(6, 16, 6*16).transpose(2, 0,
                                               1).reshape(6*16, 6*16).transpose()
    plt.figure()
    plt.imshow(test, cmap='gray')
    plt.title('comp recalée')

plt.figure()
plt.imshow(ref_img, cmap="gray")
plt.title("ref")

plt.figure()
plt.imshow(comp_image, cmap="gray")
plt.title("comp")

# %%


def complex_array_to_rgb(X, theme='dark', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y


final_alignments[np.isnan(final_alignments)] = 0
C = final_alignments[0, :, :, 0] + 1j*final_alignments[0, :, :, 1]
C = C[::2, ::2]

plt.figure()
plt.imshow(complex_array_to_rgb(C))

# %%

x = np.linspace(-5, 5, 11)
y = np.linspace(-5, 5, 11)

meshx, meshy = np.meshgrid(x, y)

plt.figure('legend')
plt.imshow(complex_array_to_rgb(meshx+1j*meshy))
