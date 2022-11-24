# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:19:47 2022

@author: jamyl
"""

from numba import cuda, float32, typeof
import ctypes
import numpy as np
import torch
import time


d_arr = cuda.device_array((4000, 5000), np.float32)
# d_arr = cuda.to_device(np.array([[10,20,30],[40,50,60.0]], dtype=np.float32))
cuda.synchronize()
t1 = time.time()
print("d_arr written on GPU")
print(d_arr.__cuda_array_interface__)

#%%
# https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda
# https://github.com/pytorch/pytorch/blob/master/test/test_numba_integration.py
# Possibilities : 
    
# tensor = torch.Tensor(d_arr) : trash, because is copies GPU numba -> CPU -> GPU torch
# tensor = torch.cuda.xTensor(d_arr) : meh, because is copies GPU numba -> GPU torch (still much faster)
# tensor = torch.as_tensor(d_arr, device="cuda") : officialy recommended by the numba team, doesnt seem to copy anything.

tensor = torch.as_tensor(d_arr, device="cuda")
tensor = tensor.type(torch.complex128) + 1j
torch.cuda.synchronize('cuda')
print("d_arr retrieved by torch")
print(time.time() - t1)
print(tensor.device)
print(tensor.__cuda_array_interface__)


#%%
numba_tensor = cuda.as_cuda_array(tensor.real.type(torch.float32))
cuda.synchronize()
print("d_arr retrieved by numba")
print(time.time() - t1)
print(numba_tensor)