# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:48 2022

@author: jamyl
"""


import numpy as np

from numba import uint8, uint16, float32, float64, cuda


@cuda.jit(device=True)
def compute_rgb(neighborhood_idy, neighborhood_idx,
                subtile_center_idy, subtile_center_idx,
                neighborhood, rgb, acc):

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 0 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 0):  # Bayer 00
        if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # b
            rgb[2] += neighborhood[neighborhood_idy, neighborhood_idx]
            acc[2] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # g top
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # g bottom
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx - 1] != np.inf:  # g left
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx - 1]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx + 1] != np.inf:  # g right
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx + 1]
            acc[1] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # r top left
            rgb[0] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[0] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # r top right
            rgb[0] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[0] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # r bottom left
            rgb[0] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[0] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # r bottom right
            rgb[0] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[0] += 1

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 0 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 1):  # Bayer 01
        if neighborhood[neighborhood_idy, neighborhood_idx-1] != np.inf:  # b left
            rgb[2] += neighborhood[neighborhood_idy, neighborhood_idx-1]
            acc[2] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx+1] != np.inf:  # b right
            rgb[2] += neighborhood[neighborhood_idy, neighborhood_idx+1]
            acc[2] += 1

        if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # g
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # g top left
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # g top right
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # g bottom left
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # g bottom right
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[1] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # r top
            rgb[0] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[0] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # r bottom
            rgb[0] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[0] += 1

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 1 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 0):  # Bayer 10
        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # b top
            rgb[2] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[2] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # b bottom
            rgb[2] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[2] += 1

        if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # g
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # g top left
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # g top right
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # g bottom left
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # g bottom right
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[1] += 1

        if neighborhood[neighborhood_idy, neighborhood_idx-1] != np.inf:  # r left
            rgb[0] += neighborhood[neighborhood_idy, neighborhood_idx-1]
            acc[0] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx+1] != np.inf:  # r right
            rgb[0] += neighborhood[neighborhood_idy, neighborhood_idx+1]
            acc[0] += 1

    if ((neighborhood_idy-2 + subtile_center_idy) % 2 == 1 and
            (neighborhood_idx-2 + subtile_center_idx) % 2 == 1):  # Bayer 11
        if neighborhood[neighborhood_idy-1, neighborhood_idx-1] != np.inf:  # b top left
            rgb[2] += neighborhood[neighborhood_idy-1, neighborhood_idx-1]
            acc[2] += 1
        if neighborhood[neighborhood_idy-1, neighborhood_idx+1] != np.inf:  # b top right
            rgb[2] += neighborhood[neighborhood_idy-1, neighborhood_idx+1]
            acc[2] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx-1] != np.inf:  # b bottom left
            rgb[2] += neighborhood[neighborhood_idy+1, neighborhood_idx-1]
            acc[2] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx+1] != np.inf:  # b bottom right
            rgb[2] += neighborhood[neighborhood_idy+1, neighborhood_idx+1]
            acc[2] += 1

        if neighborhood[neighborhood_idy-1, neighborhood_idx] != np.inf:  # g top
            rgb[1] += neighborhood[neighborhood_idy-1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy+1, neighborhood_idx] != np.inf:  # g bottom
            rgb[1] += neighborhood[neighborhood_idy+1, neighborhood_idx]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx - 1] != np.inf:  # g left
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx - 1]
            acc[1] += 1
        if neighborhood[neighborhood_idy, neighborhood_idx + 1] != np.inf:  # g right
            rgb[1] += neighborhood[neighborhood_idy, neighborhood_idx + 1]
            acc[1] += 1

    if neighborhood[neighborhood_idy, neighborhood_idx] != np.inf:  # r
        rgb[0] += neighborhood[neighborhood_idy, neighborhood_idx]
        acc[0] += 1
