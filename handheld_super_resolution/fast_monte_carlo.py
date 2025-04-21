# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:49:06 2023

This script is an extension of monte_carlo_simulation, optimised to run faster.
The simulation uses multiple CPU cores and focuses on the non linear parts of
the curve. Linear parts are interpolated, resulting in a considerable time gain.


@author: jamyl
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import time

n_patches = int(1e5)
n_brightness_levels = 1000

# alpha = 1.80710882e-4
# beta = 3.1937599182128e-6
TOL = 3

# While I +- tol*sigma is between 0 and 1, the linear model is considered
# to hold, and values are interpolated.
# 3 and 5 are good values  

#%%

def get_non_linearity_bound(alpha, beta, tol):
    tol_sq = tol*tol
    xmin = tol_sq/2 * (alpha + np.sqrt(tol_sq*alpha*alpha + 4*beta))
    
    xmax = (2 + tol_sq*alpha - np.sqrt((2+tol_sq*alpha)**2 - 4*(1+tol_sq*beta)))/2
    
    return xmin, xmax



def unitary_MC(alpha, beta, b):
    """
    Runs a MC scheme to estimate sigma and d for a given brightness, alpha and
    beta.

    Parameters
    ----------
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    b : float in 0, 1
        brighntess

    Returns
    -------
    diff_mean : float
        mean difference
    std_mean : float
        mean standard deviation

    """
    # create the patch
    patch = np.ones((n_patches, 3, 3)) * b

    # add noise and clip
    patch1 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch1 = np.clip(patch1, 0.0, 1.0)

    patch2 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch2 = np.clip(patch2, 0.0, 1.0)
 
    # compute statistics
    std_mean = 0.5 * np.mean((np.std(patch1, axis=(1, 2)) + np.std(patch2, axis=(1, 2))))

    curr_mean1 = np.mean(patch1, axis=(1, 2))
    curr_mean2 = np.mean(patch2, axis=(1, 2))
    diff_mean = np.mean(np.abs(curr_mean1 - curr_mean2))
    
    return diff_mean, std_mean

def regular_MC(b_array, alpha, beta):
    """
    Runs MC on multiple CPU cores

    Parameters
    ----------
    b_array : numpy array [n_brightness_levels]
        required brighntess levels.
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.

    Returns
    -------
    sigmas : numpy array [n_brightness_levels]
        Simulated std
    diffs : numpy array [n_brightness_levels]
        simulated differences

    """
    if multiprocessing.cpu_count() > 1:
        N_CPU = multiprocessing.cpu_count()-1
    else:
        N_CPU = 1
        multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=N_CPU)
    func = partial(unitary_MC, alpha, beta)
    
    
    sigmas = np.empty_like(b_array)
    diffs = np.empty_like(b_array)
    for b_, result in enumerate(tqdm(pool.imap(func, list(b_array)), total=b_array.size,desc="Brightnesses")):
        diffs[b_] = result[0]
        sigmas[b_] = result[1]
    
    pool.close()
    return sigmas, diffs

def interp_MC(b_array,
              sigma_min, sigma_max,
              diff_min, diff_max):
    """
    Interpolates the missing values for diff and and sigma, based on their
    upper and lower bounds.

    Parameters
    ----------
    b_array : TYPE
        DESCRIPTION.
    sigma_min : TYPE
        DESCRIPTION.
    sigma_max : TYPE
        DESCRIPTION.
    diff_min : TYPE
        DESCRIPTION.
    diff_max : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    norm_b = (b_array - b_array[0])/(b_array[-1] - b_array[0])
    
    sigmas_sq_lin = norm_b * (sigma_max**2 - sigma_min**2) + sigma_min**2
    diffs_sq_lin = norm_b * (diff_max**2 - diff_min**2) + diff_min**2
    
    return np.sqrt(sigmas_sq_lin[1:-1]), np.sqrt(diffs_sq_lin[1:-1])


def run_fast_MC(alpha, beta):
    """
    Computes diff and sigmas for 1001 values of brighntess, for fixed alpha and
    beta.
    
    To go faster, bound for which the probability of clipping is
    neglectible are estimated (xmin, xmax). A regular MC is applied to brightness
    outside of this range, and the rest of the points are interpolates linearly.

    Parameters
    ----------
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # t0 = time.perf_counter()
    print("Estimating noise curves ...")
    xmin, xmax = get_non_linearity_bound(alpha, beta, TOL)
    
    # these 2 indexes define the points for which the linear model is valid.
    # The regular Monte carlo is applied to these points, that are then used as reference
    # for the linear model
    imin = int(np.ceil(xmin * n_brightness_levels)) + 1
    imax = int(np.floor(xmax * n_brightness_levels)) - 1

    if imin > n_brightness_levels:
        print("Fast MC impossible, falling back to regular MC")
        return regular_MC(np.linspace(0, 1, n_brightness_levels+1), alpha, beta)

    sigmas = np.empty(n_brightness_levels+1)
    diffs = np.empty(n_brightness_levels+1)
    brigntess = np.arange(n_brightness_levels+1)/n_brightness_levels
    
    # running regular MC for non linear parts 
    nl_brigntess = np.concatenate((brigntess[:imin+1], brigntess[imax:]))
    sigma_nl, diffs_nl = regular_MC(nl_brigntess, alpha, beta)
    sigmas[:imin+1], diffs[:imin+1] = sigma_nl[:imin+1], diffs_nl[:imin+1]
    sigmas[imax:], diffs[imax:] = sigma_nl[imin+1:], diffs_nl[imin+1:]
    
    # padding using linear interpolation
    brightness_l = brigntess[imin-1:imax+2]
    
    sigmas_l ,diffs_l = interp_MC(brightness_l,
                                 sigmas[imin], sigmas[imax],
                                 diffs[imin], diffs[imax])
    sigmas[imin:imax+1] = sigmas_l
    diffs[imin:imax+1] = diffs_l
    # print('Finished! Estimated 1 noise curve in {:2f} sec'.format(time.perf_counter() - t0))



    
    # plt.figure("sigma")
    # plt.plot(np.linspace(0, 1, n_brightness_levels+1), sigmas)
    # plt.grid()
    # plt.axline((xmin, 0), (xmin, 1e-5), c='k')
    # plt.axline((xmax, 0), (xmax, 1e-5), c='k')
    # plt.ylabel('Sigma')
    # plt.xlabel('brightness')
    
    # plt.figure("Diff")
    # plt.plot(np.linspace(0, 1, n_brightness_levels+1), diffs)
    # plt.grid()
    # plt.axline((xmin, 0), (xmin, 1e-5), c='k')
    # plt.axline((xmax, 0), (xmax, 1e-5), c='k')
    # plt.ylabel('Diff')
    # plt.xlabel('brightness')
    return sigmas, diffs
    

if __name__ == "__main__":
    run_fast_MC(alpha=1.80710882e-4,
                beta=3.1937599182128e-6)

    
        