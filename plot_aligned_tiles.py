# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 22:35:41 2022

@author: jamyl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

a = np.load("C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus-python/results_test1/33TJ_20150606_224837_294/33TJ_20150606_224837_294_aligned_tiles.npy")

a = a[1]
a = a.reshape(197, 265, 16, 2, 32)
a = a.reshape(197, 265, 16, 2, 16, 2)
a = np.transpose(a, (0, 2, 1, 4, 3, 5))

a = a.reshape(197*16, 265*16, 2, 2)

b = np.zeros((197*16, 265*16, 3))
b[:, :, 0] = a[:, :, 1, 1]  # r
b[:, :, 1] = 1/2*(a[:, :, 1, 0] + a[:, :, 0, 1])  # g
b[:, :, 2] = a[:, :, 0, 0]  # b

#a = np.mean(a, axis = (2, 3))


plt.imshow(b/255)
