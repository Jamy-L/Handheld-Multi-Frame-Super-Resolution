# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:49:06 2023

@author: jamyl
"""

import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

n_patches = int(1e5)
n_brightness_levels = 1000

alpha = 1.80710882e-4  # from measurements
beta = 3.1937599182128e-6  # from https://www.photonstophotos.net/Charts/RN_ADU.htm chart
tol = 3

#%%

def get_non_linearity_bound(alpha, beta, tol):
    tol_sq = tol*tol
    xmin = tol_sq/2 * (alpha + np.sqrt(tol_sq*alpha*alpha + 4*beta))
    
    xmax = (2 + tol_sq*alpha - np.sqrt((2+tol_sq*alpha)**2 - 4*(1+tol_sq*beta)))/2
    
    return xmin, xmax



def unitary_MC(b):
    # create the patch
    patch = np.ones((n_patches, 3, 3)) * b

    # add noise and clip
    patch1 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch1 = np.clip(patch1, 0.0, 1.0)

    patch2 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch2 = np.clip(patch2, 0.0, 1.0)
 
    # compute statistics and store
    std_mean = 0.5 * np.mean((np.std(patch1, axis=(1, 2)) + np.std(patch2, axis=(1, 2))))

    curr_mean1 = np.mean(patch1, axis=(1, 2))
    curr_mean2 = np.mean(patch2, axis=(1, 2))
    diff_mean = np.mean(np.abs(curr_mean1 - curr_mean2))
    
    return diff_mean, std_mean

def regular_MC(b_array):
    sigmas = np.empty_like(b_array)
    diffs = np.empty_like(b_array)
    for i in tqdm(range(b_array.size)):
        sigmas[i], diffs[i] = unitary_MC(b_array[i])
    return sigmas, diffs

def interp_MC(b_array,
              sigma_min, sigma_max,
              diff_min, diff_max):
    norm_b = (b_array - b_array[0])/(b_array[-1] - b_array[0])
    
    sigmas_sq_lin = norm_b * (sigma_max**2 - sigma_min**2) + sigma_min**2
    diffs_sq_lin = norm_b * (diff_max**2 - diff_min**2) + diff_min**2
    
    return np.sqrt(sigmas_sq_lin[1:-1]), np.sqrt(diffs_sq_lin[1:-1])


def run_fast_MC(): 
    xmin, xmax = get_non_linearity_bound(alpha, beta, tol)
    
    # these 2 inddexes define the points for which the linear model is valid.
    # The regular Monte carlo is applied to these points, that are then used as reference
    # for the linear model
    imin = int(np.ceil(xmin * n_brightness_levels)) + 1
    imax = int(np.floor(xmax * n_brightness_levels)) - 1


    sigmas = np.empty(n_brightness_levels+1)
    diffs = np.empty(n_brightness_levels+1)
    brigntess = np.arange(n_brightness_levels+1)/n_brightness_levels
    
    # running regular MC for non linear parts 
    nl_brigntess = np.concatenate((brigntess[:imin+1], brigntess[imax:]))
    sigma_nl, diffs_nl = regular_MC(nl_brigntess)
    sigmas[:imin+1], diffs[:imin+1] = sigma_nl[:imin+1], diffs_nl[:imin+1]
    sigmas[imax:], diffs[imax:] = sigma_nl[imin+1:], diffs_nl[imin+1:]
    
    # padding using linear interpolation
    brightness_l = brigntess[imin-1:imax+2]
    
    sigmas_l ,diffs_l = interp_MC(brightness_l,
                                 sigmas[imin], sigmas[imax],
                                 diffs[imin], diffs[imax])
    sigmas[imin:imax+1] = sigmas_l
    diffs[imin:imax+1] = diffs_l




    
    plt.figure("sigma")
    plt.plot(np.linspace(0, 1, n_brightness_levels+1), sigmas**2)
    plt.grid()
    plt.axline((xmin, 0), (xmin, 1e-5), c='k')
    plt.axline((xmax, 0), (xmax, 1e-5), c='k')
    plt.ylabel('Sigma**2')
    plt.xlabel('brightness')
    
    plt.figure("Diff**2")
    plt.plot(np.linspace(0, 1, n_brightness_levels+1), diffs**2)
    plt.grid()
    plt.axline((xmin, 0), (xmin, 1e-5), c='k')
    plt.axline((xmax, 0), (xmax, 1e-5), c='k')
    plt.ylabel('Diff**2')
    plt.xlabel('brightness')
    


run_fast_MC()

        
        
        