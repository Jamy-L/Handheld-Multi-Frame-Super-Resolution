import math

import numpy as np
from numba import cuda
import torch
import torch.nn.functional as F

from .utils import clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS
FLOAT = DEFAULT_NUMPY_FLOAT_TYPE
SOBEL_Y = torch.as_tensor(np.array([[-1], [0], [1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_X = torch.as_tensor(np.array([[-1,0,1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_Y.requires_grad = False
SOBEL_X.requires_grad = False

def init_ica(image, tile_size, config):
    imsize_y, imsize_x = image.shape
    n_patch_y = imsize_y // tile_size 
    n_patch_x = imsize_x // tile_size

    gradx = F.conv2d(image[None, None], SOBEL_X, padding='same').squeeze()
    grady = F.conv2d(image[None, None], SOBEL_Y, padding='same').squeeze()

    gradx = cuda.as_cuda_array(gradx)
    grady = cuda.as_cuda_array(grady)

    hessian = cuda.device_array((n_patch_y, n_patch_x, 2, 2), DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(n_patch_x / threadsperblock[1])
    blockspergrid_y = math.ceil(n_patch_y / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    compute_hessian[blockspergrid, threadsperblock](gradx, grady, tile_size, hessian)

    return gradx, grady, hessian

@cuda.jit()
def compute_hessian(gradx, grady, tile_size, hessian):
    patch_idx, patch_idy = cuda.grid(2)
    n_patchy, n_patch_x, _, _ = hessian.shape
    
    # discarding non existing patches
    if not (patch_idy < n_patchy and
            patch_idx < n_patch_x):
        return
    
    patch_pos_idx = tile_size * patch_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    patch_pos_idy = tile_size * patch_idy
    
    local_hessian = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    local_hessian[0, 0] = 0
    local_hessian[0, 1] = 0
    local_hessian[1, 0] = 0
    local_hessian[1, 1] = 0
    
    for i in range(tile_size):
        for j in range(tile_size):
            pixel_global_idy = patch_pos_idy + i
            pixel_global_idx = patch_pos_idx + j

            # I think this check useless, the check above should be enough            
            # if not (0 <= pixel_global_idy < imshape[0] and 
            #         0 <= pixel_global_idx < imshape[1]):
            #     continue

            local_gradx = gradx[pixel_global_idy, pixel_global_idx]
            local_grady = grady[pixel_global_idy, pixel_global_idx]
            
            local_hessian[0, 0] += local_gradx*local_gradx
            local_hessian[0, 1] += local_gradx*local_grady
            local_hessian[1, 0] += local_gradx*local_grady
            local_hessian[1, 1] += local_grady*local_grady
                
    hessian[patch_idy, patch_idx, 0, 0] = local_hessian[0, 0]
    hessian[patch_idy, patch_idx, 0, 1] = local_hessian[0, 1]
    hessian[patch_idy, patch_idx, 1, 0] = local_hessian[1, 0]
    hessian[patch_idy, patch_idx, 1, 1] = local_hessian[1, 1]

def align_lvl_ica(ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
                  moving_lvl, alignment, l, config):
    verbose_3 = config.verbose >= 3
    tile_size = config.block_matching.tuning.tile_sizes[l]

    np_y, np_x, _ = alignment.shape

    # New way, 1 thread/pixel
    threadsperblock = (tile_size, tile_size)
    blockspergrid_x = np_x
    blockspergrid_y = np_y
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    if tile_size == 8:
        cuda_kernel = ica_kernel_8
    elif tile_size == 16:
        cuda_kernel = ica_kernel_16
    elif tile_size == 32:
        cuda_kernel = ica_kernel_32
    elif tile_size == 64:
        cuda_kernel = ica_kernel_64
        threadsperblock = (64, 16)  # because each thread handles 4 pixels
    else:
        raise NotImplementedError("ICA kernel for tile size {} not implemented".format(tile_size))
    cuda_kernel[blockspergrid, threadsperblock](
        ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
        moving_lvl, alignment, config.ica.tuning.n_iter)

@cuda.jit
def ica_kernel_8(ref_img, gradx, grady, hessian, moving, alignment, niter):
    # 1 thread/pixel, 1 block/patch
    TILE_SIZE = 8
    h, w = moving.shape
    np_y, np_x, _ = alignment.shape
    x, y  = cuda.grid(2)
    px = cuda.blockIdx.x
    py = cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tid = ty * TILE_SIZE + tx

    # x,y are inbound by design
    A00 = hessian[py, px, 0, 0]
    A01 = hessian[py, px, 0, 1]
    A10 = hessian[py, px, 1, 0]
    A11 = hessian[py, px, 1, 1]

    det = A00 * A11 - A01 * A10
    if abs(det) < 1e-10: # system is Not solvable
        return  # 1 Hessian per block, so all threads exit
    det_inv = 1.0 / det

    s_alignment = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_alignment[0] = alignment[py, px, 0]
        s_alignment[1] = alignment[py, px, 1]

    l_gradx = gradx[y, x]
    l_grady = grady[y, x]

    s_B0 = cuda.shared.array((8 * 8), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    s_B1 = cuda.shared.array((8 * 8), dtype = DEFAULT_CUDA_FLOAT_TYPE)

    ref_c = ref_img[y, x]

    for _ in range(niter):
        cuda.syncthreads()
        # Warp I with W(x; p) to compute I(W(x; p))

        ## bilinear interpolation at new_x, new_y
        floor_x = x + int(s_alignment[0])
        floor_y = y + int(s_alignment[1])
        frac_x, _ = math.modf(s_alignment[0])
        frac_y, _ = math.modf(s_alignment[1])
        # Note: in theory frac_x, floor_x = math.modf(x + alignment[0]) in 1 shot. But it is surprisingly faster to compute it from s_alignment this way 

        floor_x = clamp(floor_x, 0, w - 1)
        floor_y = clamp(floor_y, 0, h - 1)

        ceil_x = clamp(floor_x + 1, 0, w - 1)
        ceil_y = clamp(floor_y + 1, 0, h - 1)

        m00 = moving[floor_y, floor_x]
        m01 = moving[floor_y, ceil_x]
        m10 = moving[ceil_y, floor_x]
        m11 = moving[ceil_y, ceil_x]

        lerpx_top = m00 + (m01 - m00) * frac_x
        lerpx_bot = m10 + (m11 - m10) * frac_x
        mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y

        gradt = mov_interp - ref_c

        ##### Reduce within the block (=sum)
        # The reduce methods using shfl on warps is slower than this, idk why
        s_B0[tid] = -l_gradx * gradt
        s_B1[tid] = -l_grady * gradt
        N = TILE_SIZE * TILE_SIZE // 2
        while N > 0:
            cuda.syncthreads()
            if tid < N:
                s_B0[tid] += s_B0[tid + N]
                s_B1[tid] += s_B1[tid + N]
            N = N // 2
        #############

        if tid == 0:
            B0 = s_B0[0]
            B1 = s_B1[0]

            # solve Ax = B
            s_alignment[0] += det_inv * (A11 * B0 - A01 * B1)
            s_alignment[1] += det_inv * (-A10 * B0 + A00 * B1)

    if tid == 0:
        alignment[py, px, 0] = s_alignment[0]
        alignment[py, px, 1] = s_alignment[1]

@cuda.jit
def ica_kernel_16(ref_img, gradx, grady, hessian, moving, alignment, niter):
    # 1 thread/pixel, 1 block/patch
    TILE_SIZE = 16
    h, w = moving.shape
    x, y  = cuda.grid(2)
    px = cuda.blockIdx.x
    py = cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tid = ty * TILE_SIZE + tx

    # x,y are inbound by design
    A00 = hessian[py, px, 0, 0]
    A01 = hessian[py, px, 0, 1]
    A10 = hessian[py, px, 1, 0]
    A11 = hessian[py, px, 1, 1]

    det = A00 * A11 - A01 * A10
    if abs(det) < FLOAT(1e-10): # system is Not solvable
        return  # 1 Hessian per block, so all threads exit
    det_inv = FLOAT(1.0) / det

    s_alignment = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_alignment[0] = alignment[py, px, 0]
        s_alignment[1] = alignment[py, px, 1]

    l_gradx = gradx[y, x]
    l_grady = grady[y, x]

    s_B0 = cuda.shared.array((16 * 16), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    s_B1 = cuda.shared.array((16 * 16), dtype = DEFAULT_CUDA_FLOAT_TYPE)

    ref_c = ref_img[y, x]

    for _ in range(niter):
        cuda.syncthreads()
        # Warp I with W(x; p) to compute I(W(x; p))

        ## bilinear interpolation at new_x, new_y
        floor_x = x + int(s_alignment[0])
        floor_y = y + int(s_alignment[1])
        frac_x, _ = math.modf(s_alignment[0])
        frac_y, _ = math.modf(s_alignment[1])
        # Note: in theory frac_x, floor_x = math.modf(x + alignment[0]) in 1 shot. But it is surprisingly faster to compute it from s_alignment this way 

        m00 = moving[floor_y + 0, floor_x + 0] if 0 <= floor_y + 0 < h and 0 <= floor_x + 0 < w else FLOAT(0.0)
        m01 = moving[floor_y + 0, floor_x + 1] if 0 <= floor_y + 0 < h and 0 <= floor_x + 1 < w else FLOAT(0.0)
        m10 = moving[floor_y + 1, floor_x + 0] if 0 <= floor_y + 1 < h and 0 <= floor_x + 0 < w else FLOAT(0.0)
        m11 = moving[floor_y + 1, floor_x + 1] if 0 <= floor_y + 1 < h and 0 <= floor_x + 1 < w else FLOAT(0.0)

        lerpx_top = m00 + (m01 - m00) * frac_x
        lerpx_bot = m10 + (m11 - m10) * frac_x
        mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y

        gradt = mov_interp - ref_c

        ##### Reduce within the block (=sum)
        # The reduce methods using shfl on warps is slower than this, idk why
        s_B0[tid] = -l_gradx * gradt
        s_B1[tid] = -l_grady * gradt
        N = TILE_SIZE * TILE_SIZE // 2
        while N > 0:
            cuda.syncthreads()
            if tid < N:
                s_B0[tid] += s_B0[tid + N]
                s_B1[tid] += s_B1[tid + N]
            N = N // 2
        #############

        if tid == 0:
            B0 = s_B0[0]
            B1 = s_B1[0]

            # solve Ax = B
            s_alignment[0] += det_inv * (A11 * B0 - A01 * B1)
            s_alignment[1] += det_inv * (-A10 * B0 + A00 * B1)

    if tid == 0:
        alignment[py, px, 0] = s_alignment[0]
        alignment[py, px, 1] = s_alignment[1]

@cuda.jit
def ica_kernel_32(ref_img, gradx, grady, hessian, moving, alignment, niter):
    # 1 thread/pixel, 1 block/patch
    # No use of shared mem for interpolated patch because it just makes it slower
    TILE_SIZE = 32
    N_THREADS = 32 * 32
    h, w = moving.shape
    x, y  = cuda.grid(2)
    px = cuda.blockIdx.x
    py = cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tid = ty * TILE_SIZE + tx

    # x,y are inbound by design
    A00 = hessian[py, px, 0, 0]
    A01 = hessian[py, px, 0, 1]
    A10 = hessian[py, px, 1, 0]
    A11 = hessian[py, px, 1, 1]

    det = A00 * A11 - A01 * A10
    if abs(det) < FLOAT(1e-10): # system is Not solvable
        return  # 1 Hessian per block, so all threads exit
    det_inv = FLOAT(1.0) / det

    s_alignment = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_alignment[0] = alignment[py, px, 0]
        s_alignment[1] = alignment[py, px, 1]

    l_gradx = gradx[y, x]
    l_grady = grady[y, x]

    s_B0 = cuda.shared.array((32 * 32), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    s_B1 = cuda.shared.array((32 * 32), dtype = DEFAULT_CUDA_FLOAT_TYPE)

    ref_c = ref_img[y, x]

    for _ in range(niter):
        cuda.syncthreads() # Wait for s_alignment to be ready
        # Warp I with W(x; p) to compute I(W(x; p))

        ## bilinear interpolation at new_x, new_y
        frac_x, _ = math.modf(s_alignment[0])
        frac_y, _ = math.modf(s_alignment[1])

        floor_x = x + int(s_alignment[0])
        floor_y = y + int(s_alignment[1])

        m00 = moving[floor_y + 0, floor_x + 0] if 0 <= floor_y + 0 < h and 0 <= floor_x + 0 < w else FLOAT(0.0)
        m01 = moving[floor_y + 0, floor_x + 1] if 0 <= floor_y + 0 < h and 0 <= floor_x + 1 < w else FLOAT(0.0)
        m10 = moving[floor_y + 1, floor_x + 0] if 0 <= floor_y + 1 < h and 0 <= floor_x + 0 < w else FLOAT(0.0)
        m11 = moving[floor_y + 1, floor_x + 1] if 0 <= floor_y + 1 < h and 0 <= floor_x + 1 < w else FLOAT(0.0)

        lerpx_top = m00 + (m01 - m00) * frac_x
        lerpx_bot = m10 + (m11 - m10) * frac_x
        mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y

        gradt = mov_interp - ref_c
        B0 = -l_gradx * gradt
        B1 = -l_grady * gradt

        ##### Reduce within the block (=sum)
        WARP_SIZE = 32
        WARPS_PER_BLOCK = N_THREADS // WARP_SIZE
        offset = WARP_SIZE // 2
        # Sum thread among warps first
        while offset > 0:
            B0 += cuda.shfl_down_sync(0xffffffff, B0, offset)
            B1 += cuda.shfl_down_sync(0xffffffff, B1, offset)
            offset //= 2
        if tid % WARP_SIZE == 0:
            s_B0[tid//WARP_SIZE] = B0
            s_B1[tid//WARP_SIZE] = B1
        cuda.syncthreads() # Wait for warp sums to be ready

        if tid == 0:
            B0 = s_B0[0]
            B1 = s_B1[0]
            for i in range(1, WARPS_PER_BLOCK):
                B0 += s_B0[i]
                B1 += s_B1[i]

            # solve Ax = B
            s_alignment[0] += det_inv * (A11 * B0 - A01 * B1)
            s_alignment[1] += det_inv * (-A10 * B0 + A00 * B1)

    if tid == 0:
        alignment[py, px, 0] = s_alignment[0]
        alignment[py, px, 1] = s_alignment[1]

@cuda.jit
def ica_kernel_64(ref_img, gradx, grady, hessian, moving, alignment, niter):
    ### 1 thread handles 4 pixels vertically, 1 block/patch
    # The reads from global mem are suprisingly well coalesced, so moving to shared mem would just waste time
    # - Each thread compute its 4 interpolated pixels and the corresponding B0,B1
    # - Each thread accumulates the 4 B0,B1 in local registers. 16 * 64 values for B0 remain to accumulate
    # - B0 and B1 are then reduced per warp (one row chunk of 32 pixels). 32 values remain to accumulate (1 per warp)
    # - Each warp writes its accumulated B0,B1 in shared memory.
    # - Finally, the 32 values are sumed using a single thread B0,B1 (multithreaded reduction would take slightly longer)
    # - Thread 0 does the inversion and updates the alignments, ready for next iteration

    TILE_SIZE = 64
    N_THREADS = 16 * 64
    h, w = moving.shape
    px = cuda.blockIdx.x
    py = cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    ti = ty * cuda.blockDim.x + tx
    x = px * TILE_SIZE + tx
    y = py * TILE_SIZE + ty * 4  # each thread handles 4 vertical pixels

    A00 = hessian[py, px, 0, 0]
    A01 = hessian[py, px, 0, 1]
    A10 = hessian[py, px, 1, 0]
    A11 = hessian[py, px, 1, 1]

    det = A00 * A11 - A01 * A10
    if abs(det) < FLOAT(1e-10): # system is Not solvable
        return  # 1 Hessian per block, so all threads exit
    det_inv = FLOAT(1.0) / det

    s_alignment = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    if ti == 0:
        s_alignment[0] = alignment[py, px, 0]
        s_alignment[1] = alignment[py, px, 1]

    # Specific to 64x64: patchs: the threads manages 4 pixels, so we store 4 gradients
    l_gradx = cuda.local.array(4, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    l_grady = cuda.local.array(4, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    l_gradx[0] = gradx[y + 0, x]
    l_grady[0] = grady[y + 0, x]
    l_gradx[1] = gradx[y + 1, x]
    l_grady[1] = grady[y + 1, x]
    l_gradx[2] = gradx[y + 2, x]
    l_grady[2] = grady[y + 2, x]
    l_gradx[3] = gradx[y + 3, x]
    l_grady[3] = grady[y + 3, x]
    
    # One cell per warp (there are 32 warps in the block)
    s_B0 = cuda.shared.array((32), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    s_B1 = cuda.shared.array((32), dtype = DEFAULT_CUDA_FLOAT_TYPE)

    ref_c = cuda.local.array(4, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    ref_c[0] = ref_img[y + 0, x]
    ref_c[1] = ref_img[y + 1, x]
    ref_c[2] = ref_img[y + 2, x]
    ref_c[3] = ref_img[y + 3, x]

    for _ in range(niter):
        cuda.syncthreads() # Wait for s_alignment to be ready
        # Warp I with W(x; p) to compute I(W(x; p))

        B0 = FLOAT(0.0)
        B1 = FLOAT(0.0)

        ## bilinear interpolation at new_x, new_y
        frac_x, _ = math.modf(s_alignment[0])
        frac_y, _ = math.modf(s_alignment[1])
        floor_x = x + int(s_alignment[0])

        # We need to interpolate 4 vertically aligned pixels, so we do it with a sliding strategy ; reuse lerpx as we slide down
        floor_y = y + int(s_alignment[1])
        m10 = moving[floor_y, floor_x + 0] if 0 <= floor_y < h and 0 <= floor_x + 0 < w else FLOAT(0.0)
        m11 = moving[floor_y, floor_x + 1] if 0 <= floor_y < h and 0 <= floor_x + 1 < w else FLOAT(0.0)
        lerpx_bot = m10 + (m11 - m10) * frac_x
        for i in range(4):
            floor_y += 1

            m10 = moving[floor_y + 1, floor_x + 0] if 0 <= floor_y + 1 < h and 0 <= floor_x + 0 < w else FLOAT(0.0)
            m11 = moving[floor_y + 1, floor_x + 1] if 0 <= floor_y + 1 < h and 0 <= floor_x + 1 < w else FLOAT(0.0)

            lerpx_top = lerpx_bot
            lerpx_bot = m10 + (m11 - m10) * frac_x
            mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y

            gradt = mov_interp - ref_c[i]

            B0 += -l_gradx[i] * gradt
            B1 += -l_grady[i] * gradt

        WARP_SIZE = 32
        WARPS_PER_BLOCK = N_THREADS // WARP_SIZE
        offset = WARP_SIZE // 2
        # Sum thread among warps first
        while offset > 0:
            B0 += cuda.shfl_down_sync(0xffffffff, B0, offset)
            B1 += cuda.shfl_down_sync(0xffffffff, B1, offset)
            offset //= 2


        if ti % WARP_SIZE == 0 and ti > 0: # No need to write the first warp value, because warp0 takes care of the final sum
            s_B0[ti//WARP_SIZE] = B0
            s_B1[ti//WARP_SIZE] = B1
        cuda.syncthreads() # Wait for warp sums to be ready
        # Then sum warps
        if ti == 0:
            for i in range(1, WARPS_PER_BLOCK):
                B0 += s_B0[i]
                B1 += s_B1[i]

            # solve Ax = B
            s_alignment[0] += det_inv * (A11 * B0 - A01 * B1)
            s_alignment[1] += det_inv * (-A10 * B0 + A00 * B1)

    if ti == 0:
        alignment[py, px, 0] = s_alignment[0]
        alignment[py, px, 1] = s_alignment[1]