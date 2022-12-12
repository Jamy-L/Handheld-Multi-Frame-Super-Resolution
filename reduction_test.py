# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:56:46 2022

@author: jamyl
"""
from tqdm import tqdm
import numpy as np
from numba import cuda, float32, float64
from math import log2
import time
import matplotlib.pyplot as plt




threads = 32
threads_sq = 32*32

radius = 4
sR = 2*radius + 1

h, w = 125, 94
imshape = (1532, 2028)

blocks = (w, h, sR*sR)


#%% convolution test


@cuda.jit # (cache=True)
def naive_convolve(cu_image, cu_kernel, output):
    tx, ty = cuda.threadIdx.x,cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    
    idx = bx + tx
    idy = by + ty
    
    s = 0
    if tx == 0 and ty ==0:
        for i in range(16):
            for j in range(16):
                a = cu_image[idy + i, idx + j]
                b = cu_kernel[i, j]
                s += a*b
        output[by, bx] = s


    
@cuda.jit # (cache=True)
def fancy_convolve(cu_image, cu_kernel, output):
    tx, ty = cuda.threadIdx.x,cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    
    idx = bx + tx
    idy = by + ty

    
    z = int(tx + ty*16)
    s = cuda.shared.array(32*32, float64)
    
    s[z] = cu_image[idy, idx]*cu_kernel[ty, tx]
    
    N_reduc = int(log2(16*16))
    # reduction
    step = 1
    for reduc in range(N_reduc):
        cuda.syncthreads()
        if tx%(2*step) == 0:
            s[tx] += s[tx + step]
        
        step *= 2
    
    cuda.syncthreads()
    if tx == 0 and ty==0:
        output[by, bx] = s[0]



@cuda.jit # (cache=True)
def shared_mem_convolve(cu_image, cu_kernel, output):
    tx, ty = cuda.threadIdx.x,cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    
    idx = bx + tx
    idy = by + ty

    ref_patch = cuda.shared.array((32, 32), float64)
    sh_kernel = cuda.shared.array((32, 32), float64)
    
    ref_patch[ty, tx] = cu_image[idy, idx]
    sh_kernel[ty, tx] = cu_kernel[ty, tx]
    
    cuda.syncthreads()
    
    
    z = int(tx + ty*16)
    s = cuda.shared.array(32*32, float64)
    
    s[z] = ref_patch[ty, tx]*sh_kernel[ty, tx]
    
    N_reduc = int(log2(16*16))
    # reduction
    step = 1
    for reduc in range(N_reduc):
        cuda.syncthreads()
        if tx%(2*step) == 0:
            s[tx] += s[tx + step]
        
        step *= 2
    
    cuda.syncthreads()
    if tx == 0 and ty==0:
        output[by, bx] = s[0]

@cuda.jit # (cache=True)
def atomic_convolve(cu_image, cu_kernel, output):
    tx, ty = cuda.threadIdx.x,cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    
    idx = bx + tx
    idy = by + ty

    s = cuda.shared.array(1, float64)
    if tx == 0 and ty == 0:
        s[0] = 0
    cuda.syncthreads()
    
    cuda.atomic.add(s, 0, cu_image[idy, idx]*cu_kernel[ty, tx])
    
    cuda.syncthreads()
    if tx == 0 and ty==0:
        output[by, bx] = s[0]


#### Distances L1

# note that the naive implementation is using shared mem, and cannot even
# be launched if it is too naive !
@cuda.jit
def naive_L1Distance(referencePyramidLevel, alternatePyramidLevel,
                     tileSize, searchRadius, dst):
    
    tile_x, tile_y, z = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    x, y = cuda.threadIdx.x, cuda.threadIdx.y
    sR = 2*searchRadius+1
    sRx = int(z//sR)
    sRy = int(z%sR)
    
    # no flow for now
    local_flow = cuda.local.array(2, float64)
    local_flow[0] = 0 #upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
    local_flow[1] = 0 #upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
    
    idx = int(tile_x * tileSize//2 + x)
    idy = int(tile_y * tileSize//2 + y)
    new_idx = int(idx + local_flow[0] + sRx - searchRadius)
    new_idy = int(idy + local_flow[1] + sRy - searchRadius)
    
    ref_patch = cuda.shared.array((32, 32), float64)
    alt_patch = cuda.shared.array((32, 32), float64)
    if (0 <= new_idx < referencePyramidLevel.shape[1] and
        0 <= new_idy < referencePyramidLevel.shape[0]) :
        ref_patch[y, x] = referencePyramidLevel[idy, idx]
        alt_patch[y, x] = alternatePyramidLevel[new_idy, new_idx]
    cuda.syncthreads()
    
    
    if x == 0 and y == 0:
        s = 0
        for tx in range(tileSize):
            for ty in range(tileSize):
                idx = int(tile_x * tileSize//2 + tx)
                idy = int(tile_y * tileSize//2 + ty)
            
                new_idx = int(idx + local_flow[0] + sRx - searchRadius)
                new_idy = int(idy + local_flow[1] + sRy - searchRadius)
                
                if not (0 <= new_idx < referencePyramidLevel.shape[1] and
                        0 <= new_idy < referencePyramidLevel.shape[0]) :
                    local_dst = 1/0 # infty out of bound
                else:
                    local_dst = abs(ref_patch[ty, tx] - alt_patch[ty, tx])
                s += local_dst
                

        dst[tile_y, tile_x, sRy, sRx] = s


@cuda.jit
def atomic_L1Distance(referencePyramidLevel, alternatePyramidLevel,
                            tileSize, searchRadius, dst):
    
    tile_x, tile_y, z = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    sR = 2*searchRadius+1
    sRx = int(z//sR)
    sRy = int(z%sR)
    
    idx = int(tile_x * tileSize//2 + tx)
    idy = int(tile_y * tileSize//2 + ty)
    
    # no flow for now
    local_flow = cuda.shared.array(2, float64)
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y <= 1:
        local_flow[cuda.threadIdx.y] = 0 #upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
    cuda.syncthreads()
    
    new_idx = int(idx + local_flow[0] + sRx - searchRadius)
    new_idy = int(idy + local_flow[1] + sRy - searchRadius)
    
    d = cuda.shared.array(1, float64)
    
    if tx == 0 and ty==0:
        d[0]=0
    cuda.syncthreads()
    
    if not (0 <= new_idx < referencePyramidLevel.shape[1] and
            0 <= new_idy < referencePyramidLevel.shape[0]) :
        local_dst = 1/0 # infty out of bound
    else:
        local_dst = abs(referencePyramidLevel[idy, idx]-alternatePyramidLevel[new_idy, new_idx])
    cuda.atomic.add(d, 0,local_dst)
    
    cuda.syncthreads()
    
    if tx==0 and ty ==0:
        dst[tile_y, tile_x, sRy, sRx] = d[0]
    
@cuda.jit
def fancy_L1Distance(referencePyramidLevel, alternatePyramidLevel,
                            tileSize, searchRadius, dst):
    
    tile_x, tile_y, z = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    sR = 2*searchRadius+1
    sRx = int(z//sR)
    sRy = int(z%sR)
    
    idx = int(tile_x * tileSize//2 + tx)
    idy = int(tile_y * tileSize//2 + ty)
    
    # no flow for now
    local_flow = cuda.shared.array(2, float64)
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y <= 1:
        local_flow[cuda.threadIdx.y] = 0 #upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
    cuda.syncthreads()
    
    new_idx = int(idx + local_flow[0] + sRx - searchRadius)
    new_idy = int(idy + local_flow[1] + sRy - searchRadius)
    
    # 32x32 is the max size. We may need less, but shared array size must
    # be known at compilation time. working with a flattened array makes reduction
    # easier
    d = cuda.shared.array((32*32), np.float64)
    
    z = tx + ty*tileSize # flattened id
    if not (0 <= new_idx < referencePyramidLevel.shape[1] and
            0 <= new_idy < referencePyramidLevel.shape[0]) :
        local_diff = 1/0 # infty out of bound
    else :
        local_diff = referencePyramidLevel[idy, idx]-alternatePyramidLevel[new_idy, new_idx]
        
    d[z] = abs(local_diff)
    
    
    # sum reduction
    N_reduction = int(log2(tileSize**2))
    
    step = 1
    for i in range(N_reduction):
        cuda.syncthreads()
        if z%(2*step) == 0:
            d[z] += d[z + step]
        
        step *= 2
    
    cuda.syncthreads()
    if tx== 0 and ty==0:
        dst[tile_y, tile_x, sRy, sRx] = d[0]
        
@cuda.jit
def shared_L1Distance(referencePyramidLevel, alternatePyramidLevel,
                      tileSize, searchRadius, dst):
    
    tile_x, tile_y, z = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    sR = 2*searchRadius+1
    sRx = int(z//sR)
    sRy = int(z%sR)
    
    idx = int(tile_x * tileSize//2 + tx)
    idy = int(tile_y * tileSize//2 + ty)
    
    # no flow for now
    local_flow = cuda.shared.array(2, float64)
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y <= 1:
        local_flow[cuda.threadIdx.y] = 0 #upsampledAlignments[tile_y, tile_x, cuda.threadIdx.y]
    cuda.syncthreads()
    
    new_idx = int(idx + local_flow[0] + sRx - searchRadius)
    new_idy = int(idy + local_flow[1] + sRy - searchRadius)
    
    
    ref_patch = cuda.shared.array((32, 32), float64)
    alt_patch = cuda.shared.array((32, 32), float64)
    if (0 <= new_idx < referencePyramidLevel.shape[1] and
        0 <= new_idy < referencePyramidLevel.shape[0]) :
        ref_patch[ty, tx] = referencePyramidLevel[idy, idx]
        alt_patch[ty, tx] = alternatePyramidLevel[new_idy, new_idx]
    cuda.syncthreads()
    
    # 32x32 is the max size. We may need less, but shared array size must
    # be known at compilation time. working with a flattened array makes reduction
    # easier
    d = cuda.shared.array((32*32), np.float64)
    
    z = tx + ty*tileSize # flattened id
    if not (0 <= new_idx < referencePyramidLevel.shape[1] and
            0 <= new_idy < referencePyramidLevel.shape[0]) :
        local_diff = 1/0 # infty out of bound
    else :
        local_diff = ref_patch[ty, tx] - alt_patch[ty, tx]
        
    d[z] = abs(local_diff)
    
    
    # sum reduction
    N_reduction = int(log2(tileSize**2))
    
    step = 1
    for i in range(N_reduction):
        cuda.syncthreads()
        if z%(2*step) == 0:
            d[z] += d[z + step]
        
        step *= 2
    
    cuda.syncthreads()
    if tx== 0 and ty==0:
        dst[tile_y, tile_x, sRy, sRx] = d[0]


def benchmark_conv(functions, labels, N_iter=50):
    n_functions = len(functions)
    T = [[] for i in range(n_functions)]
    blocks = (2*radius +1, 2*radius + 1)
    for f_id, function in enumerate(functions):
        print("benchmarking {}".format(labels[f_id]))
        for i in tqdm(range(N_iter+1)):
            kernel = np.random.random((16, 16))
            image = np.random.random((16+2*radius, 16+2*radius))
            cu_output = cuda.device_array((2*radius+1, 2*radius+1), np.float64)
        
            cu_kernel = cuda.to_device(kernel)
            cu_image = cuda.to_device(image)
            
            cuda.synchronize()
            t0 = time.perf_counter()
            function[blocks, (16, 16)
                        ](cu_image, cu_kernel, cu_output)
            cuda.synchronize()
            if i >0:
                T[f_id].append(time.perf_counter() - t0)
                # skipping compilation (cache=False)
    plt.boxplot(T)
    plt.xticks([i+1 for i in range(n_functions)], labels)
    
def benchmark_dist(functions, labels, N_iter=10):
    n_functions = len(functions)
    T = [[] for i in range(n_functions)]
    
    
    threadsPerBlock = (threads, threads)
    blocks = (w, h, sR*sR)
    for f_id, function in enumerate(functions):
        print("benchmarking {}".format(labels[f_id]))
        for i in tqdm(range(N_iter+1)):
            ref = np.random.random(imshape)
            alt = np.random.random(imshape)
            dst = cuda.device_array((h, w, sR, sR), np.float64)
        
            cu_ref = cuda.to_device(ref)
            cu_alt = cuda.to_device(alt)
            
            
            # print("blocks : ", blocks)
            # print("threads : ", threads)
            # print("imshape : ", imshape)
            # print("patchs : ", w, h)
            cuda.synchronize()
            t0 = time.perf_counter()
            function[blocks, (threads, threads)
                        ](cu_ref, cu_alt, threads, radius, dst)
            cuda.synchronize()
            if i > 0:
                T[f_id].append(time.perf_counter() - t0)
                # skipping compilation (cache=False)
    plt.boxplot(T)
    plt.xticks([i+1 for i in range(n_functions)], labels)
    
# benchmark_conv([naive_convolve,
#            fancy_convolve, shared_mem_convolve, atomic_convolve],
#           ["naive convolve", "fancy convolve", "shared mem", "atomic_add"],
#           N_iter = 500)

benchmark_dist([fancy_L1Distance, shared_L1Distance],
               ["fancy", "shared"])