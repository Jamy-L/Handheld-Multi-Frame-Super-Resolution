# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:24:42 2022

@author: jamyl
"""

import torch as th
from torch.fft import fft2, ifft2
from fft_conv_pytorch import FFTConv2d
from scipy.signal import correlate2d
import numpy as np
import matplotlib.pyplot as plt

n_boxx = 3
n_boxy = 2
nx = 3
ny = 3
n_deplacement = 81
deplacement = 4

# version avec deplacement pour les normes
comp_layer_disp = 1000*th.rand(n_deplacement, n_boxx*n_boxy,  nx, ny)
n_comp = th.linalg.matrix_norm(comp_layer_disp)
# version "vanilla" concatnennée pour la conv kernel
# chaque patch est repété n_boxx*n_yy fois sur chaque coordonnée
comp_layer_kernel = 1000*th.rand(n_boxx*n_boxy, n_boxx*n_boxy, nx, ny)

# zones de recherche
# TODO Il faut penser à padder des 0 sur les zones centrées sur un bord;
# Ou tout autre méthode pour ne pas se fair avoir en sortie
ref_layer = 1000*th.rand(1, n_boxx*n_boxy,  2*deplacement+1, 2*deplacement+1)
# normes des zones de recherches croppées
n_ref = th.linalg.matrix_norm(ref_layer[:,:,(deplacement-nx//2):(deplacement+nx//2)+1,
                                 (deplacement-ny//2):(deplacement+ny//2)+1])


fft_conv = FFTConv2d(n_boxx*n_boxy,n_boxx*n_boxy,nx, bias = False,
                     stride = nx)
fft_conv.weight = th.nn.Parameter(comp_layer_kernel)
conv = fft_conv(ref_layer) #nx stride
# be careful, the return conv is square. Outsider value must be ignored

# + crop des bords sur les 2 dernières dimensions.


#%%

#comp_layer = th.ones(81, 2, 3, 3, 3)
comp_layer = 1000*th.rand(81, 3*2,3, 3)
comp_layer_full = 1000*th.rand(3*2, 3*3)
ref_layer_full = 1000*th.rand(3*2,9, 9)
# [n_pos*2, n_boxx, n_boxy, nx, ny]
n_comp = th.linalg.matrix_norm(comp_layer) ** 2
ref_layer = comp_layer[0]
# [n_boxx, n_boxy, nx, ny]
n_ref = th.linalg.matrix_norm(ref_layer) ** 2


signal = th.randn(1, 3*2, 9,9)
kernel = th.randn(3*2, 3*2, 3,3)

fft_conv = FFTConv2d(6,6,3, bias = False)
fft_conv.weight = th.nn.Parameter(kernel)
conv = fft_conv(signal) #nx stride



#%%
ref_tiles = ref_layer
comp_tiles = comp_layer[0]

fft_ref = fft2(ref_tiles, dim=(- 2, - 1), s=(5,5))
fft_comp = fft2(comp_tiles, dim=(- 2, - 1), s=(5,5))

pre_corr = fft_ref.conj().transpose(-2, -1) * fft_comp

corr = abs(ifft2(pre_corr, s=(5, 5),
              dim=(- 2, - 1)
              ))
# conv = conv.permute(2, 3, 0, 1)
# conv = conv.reshape(9*9, 2, 3)

D2 = n_ref + n_comp[0] - 2*corr[:, :, 0, 0]
print(D2)

