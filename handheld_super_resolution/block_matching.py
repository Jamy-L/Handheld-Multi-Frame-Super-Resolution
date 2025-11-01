import time
import math

import numpy as np
from numba import cuda
import torch

from .utils import clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS

BOX_FILTER_8 = torch.as_tensor(np.ones((1,1,8,8)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_8.requires_grad = False
BOX_FILTER_16 = torch.as_tensor(np.ones((1,1,16,16)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_16.requires_grad = False
BOX_FILTER_32 = torch.as_tensor(np.ones((1,1,32,32)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_32.requires_grad = False
BOX_FILTER_64 = torch.as_tensor(np.ones((1,1,64,64)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_64.requires_grad = False

def align_lvl_block_matching_L2(tyled_pyr_lvl, ref_fft_lvl, moving_lvl, alignment, l, config):
    verbose = config.verbose > 2
    currentTime = time.perf_counter()
    tileSize = config.block_matching.tuning.tile_sizes[l]
    searchRadius = config.block_matching.tuning.search_radii[l]
    distanceMetric = config.block_matching.tuning.metrics[l]

    imshape = moving_lvl.shape
    
    search_size = 2 * searchRadius + tileSize # The size of the crop in which the search is done
    corr_size = 2 * searchRadius + 1 # The size of the correlation map output

    # Extract tiles based on the optical flow
    search_area = extract_flow_patches(moving_lvl, alignment, tileSize, searchRadius)

    moving_fft = torch.fft.rfft2(search_area, dim=(-2, -1))
    corrs = torch.fft.irfft2(torch.conj(ref_fft_lvl) * moving_fft, s=(search_size, search_size))
    corrs = torch.fft.fftshift(corrs, dim=(-2, -1))

    # crop to valid region ±R around center output size search_size
    # Before cropping, corrs has an even shaped. The correlation with shift=0 is almost at the center, biased towards the bottom left. By removing 1 pixel from top and left, we center it.
    # The rest of the crop removes the phantom "circular" correlations at the borders due to the FFT
    pre_crop_size = corrs.shape[-1]
    crop = (pre_crop_size - 1 - corr_size) // 2
    corrs = corrs[..., crop+1:crop+corr_size+1, crop+1:crop+corr_size+1]


    ## Now compute the windows L2 norm of the search patches
    if tileSize == 8:
        box_filter = BOX_FILTER_8
    elif tileSize == 16:
        box_filter = BOX_FILTER_16
    elif tileSize == 32:
        box_filter = BOX_FILTER_32
    elif tileSize == 64:
        box_filter = BOX_FILTER_64
    else:
        raise NotImplementedError("Box filter for tile size {} not implemented".format(tileSize))
    
    # TODO there may be a faster and smarter way than conv2d for this, but this is not the bottleneck so far
    L2_search = torch.nn.functional.conv2d(
        search_area.view(-1, 1, search_size, search_size).square(), box_filter, padding="valid")
    L2_search = L2_search.view(search_area.shape[0], search_area.shape[1], L2_search.shape[-2], L2_search.shape[-1])

    ## Final L2 error computation (Not the full L2, but enough to find the minimum)
    L2_error = L2_search - 2 * corrs

    L2_error_ = L2_error.flatten(-2, -1)
    max_idx = torch.argmin(L2_error_, dim=-1)
    peak_y = max_idx // corr_size
    peak_x = max_idx % corr_size
    dy = peak_y - corr_size//2
    dx = peak_x - corr_size//2

    # Test alignment here
    alignment[:, :, 0] += dx
    alignment[:, :, 1] += dy
    
def align_lvl_block_matching_L1(ref_lvl, moving_lvl, alignments, l, config):
    h, w, _ = alignments.shape
    tile_size = config.block_matching.tuning.tile_sizes[l]
    search_radius = config.block_matching.tuning.search_radii[l]
    ny, nx, _ = alignments.shape

    # New way, 1 thread per pixel
    threadsperblock = (tile_size, tile_size)
    if tile_size == 8:
        raise NotImplementedError("L1 local search kernel for tile size {} not implemented".format(tile_size))
    elif tile_size == 16:
        kernel = cuda_L1_local_search16
        assert 2 * search_radius + 16 <= 32, f"L1 local search kernel only implemented for search windows up to size 32, which is not the case for tile size {tile_size} and search radius {search_radius}."
    elif tile_size == 32:
        kernel = cuda_L1_local_search32
    elif tile_size == 64:
        kernel = cuda_L1_local_search64
        threadsperblock = (32, 32)  # because each thread handles 4 pixels
    else:
        raise NotImplementedError("L1 local search kernel for tile size {} not implemented".format(tile_size))
    blockspergrid_x = nx
    blockspergrid_y = ny
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    kernel[blockspergrid, threadsperblock](
        ref_lvl, moving_lvl, search_radius, alignments)

@cuda.jit
def cuda_L1_local_search16(ref, moving, search_radius, alignments):
    TILE_SIZE = 16
    h, w = moving.shape
    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tile_x = cuda.blockIdx.x
    tile_y = cuda.blockIdx.y
    tid = ty * TILE_SIZE + tx
    
    # x and y are by design in the image.
    
    s_flow = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_flow[0] = round(alignments[tile_y, tile_x, 0])
        s_flow[1] = round(alignments[tile_y, tile_x, 1])

    # Load ref patch into shared memory
    s_ref = cuda.shared.array((16, 16), DEFAULT_CUDA_FLOAT_TYPE)
    s_ref[ty, tx] = ref[y, x]

    cuda.syncthreads()
    mov_y = y + int(s_flow[1]) - search_radius
    mov_x = x + int(s_flow[0]) - search_radius

    s_mov = cuda.shared.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE) # Only (2*search_radius + TILE_SIZE)**2 used
    s_mov[ty, tx] = moving[mov_y, mov_x] if 0 <= mov_x < w and 0 <= mov_y < h else 0.0

    # Load the remaining pixels
    if tx < 2 * search_radius:
        s_mov[ty, tx + TILE_SIZE] = moving[mov_y, mov_x + TILE_SIZE] if 0 <= mov_x + TILE_SIZE < w and 0 <= mov_y < h else 0.0
    if ty < 2 * search_radius:
        s_mov[ty + TILE_SIZE, tx] = moving[mov_y + TILE_SIZE, mov_x] if 0 <= mov_x < w and 0 <= mov_y + TILE_SIZE < h else 0.0
    if tx < 2 * search_radius and ty < 2 * search_radius:
        s_mov[ty + TILE_SIZE, tx + TILE_SIZE] = moving[mov_y + TILE_SIZE, mov_x + TILE_SIZE] if 0 <= mov_x + TILE_SIZE < w and 0 <= mov_y + TILE_SIZE < h else 0.0
    cuda.syncthreads()

    s_l1_map = cuda.shared.array(16 * 16, DEFAULT_CUDA_FLOAT_TYPE) # used to accumulate l1 between threads
    s_err = cuda.shared.array((16, 16), DEFAULT_CUDA_FLOAT_TYPE) # Stores the error map. We only use (2*search_radius + 1)**2 values
    for shift_y in range(-search_radius, search_radius + 1):
        for shift_x in range(-search_radius, search_radius + 1):
            ### Fancy reduce = sum accros threads
            l1_sum = abs(s_ref[ty, tx] - s_mov[ty + shift_y + search_radius, tx + shift_x + search_radius])
            t_per_warp = 32
            w_per_block = (TILE_SIZE * TILE_SIZE) // t_per_warp
            offset = t_per_warp // 2
            # Sum thread among warps first
            while offset > 0:
                l1_sum += cuda.shfl_down_sync(0xffffffff, l1_sum, offset)
                offset //= 2
            if tid % t_per_warp == 0:
                s_l1_map[tid] = l1_sum
            cuda.syncthreads()
            # Then sum warps
            if tid == 0:
                for w_id in range(t_per_warp, w_per_block * t_per_warp, t_per_warp):
                    s_l1_map[0] += s_l1_map[w_id]
            ###########

            if tid == 0:
                s_err[shift_y + search_radius, shift_x + search_radius] = s_l1_map[0]
    
    # Now find the minimum error and corresponding shift
    if tid == 0:
        err = s_err[0, 0]
        for i in range(2*search_radius + 1):
            for j in range(2*search_radius + 1):
                min = s_err[i, j]
                if err < min:
                    min = err
                    min_shift_y = i - search_radius
                    min_shift_x = j - search_radius


    alignments[tile_y, tile_x, 0] = s_flow[0] + min_shift_x
    alignments[tile_y, tile_x, 1] = s_flow[1] + min_shift_y

@cuda.jit
def cuda_L1_local_search32(ref, moving, search_radius, alignments):
    TILE_SIZE = 32
    h, w = moving.shape
    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tile_x = cuda.blockIdx.x
    tile_y = cuda.blockIdx.y
    tid = ty * TILE_SIZE + tx
    
    s_flow = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_flow[0] = round(alignments[tile_y, tile_x, 0])
        s_flow[1] = round(alignments[tile_y, tile_x, 1])

    # Load ref patch into shared memory
    s_ref = cuda.shared.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    s_ref[ty, tx] = ref[y, x] if (0 <= x < w and 0 <= y < h) else 0.0
    cuda.syncthreads()

    mov_y = y + int(s_flow[1])
    mov_x = x + int(s_flow[0])
    
    s_l1_map = cuda.shared.array(32 * 32, DEFAULT_CUDA_FLOAT_TYPE)
    s_err = cuda.shared.array((16, 16), DEFAULT_CUDA_FLOAT_TYPE)
    for shift_y in range(-search_radius, search_radius + 1):
        for shift_x in range(-search_radius, search_radius + 1):
            ### Fancy reduce = sum across threads
            l1_sum = abs(s_ref[ty, tx] - moving[mov_y + shift_y, mov_x + shift_x]) if 0 <= mov_x + shift_x < w and 0 <= mov_y + shift_y < h else 1/0
            t_per_warp = 32
            w_per_block = (TILE_SIZE * TILE_SIZE) // t_per_warp
            offset = t_per_warp // 2
            # Sum thread among warps first
            while offset > 0:
                l1_sum += cuda.shfl_down_sync(0xffffffff, l1_sum, offset)
                offset //= 2
            if tid % t_per_warp == 0:
                s_l1_map[tid] = l1_sum
            cuda.syncthreads()
            # Then sum warps
            if tid == 0:
                for w_id in range(t_per_warp, w_per_block * t_per_warp, t_per_warp):
                    s_l1_map[0] += s_l1_map[w_id]
            ###########

            
            if tid == 0:
                s_err[shift_y + search_radius, shift_x + search_radius] = s_l1_map[0]
    
    # Now find the minimum error and corresponding shift
    if tid == 0:
        err = s_err[0, 0]
        for i in range(2*search_radius + 1):
            for j in range(2*search_radius + 1):
                min = s_err[i, j]
                if err < min:
                    min = err
                    min_shift_y = i - search_radius
                    min_shift_x = j - search_radius


    alignments[tile_y, tile_x, 0] = s_flow[0] + min_shift_x
    alignments[tile_y, tile_x, 1] = s_flow[1] + min_shift_y

@cuda.jit
def cuda_L1_local_search64(ref, moving, search_radius, alignments):
    TILE_SIZE = 64
    h, w = moving.shape
    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tile_x = cuda.blockIdx.x
    tile_y = cuda.blockIdx.y
    tid = ty * cuda.blockDim.x + tx

    s_flow = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_flow[0] = round(alignments[tile_y, tile_x, 0])
        s_flow[1] = round(alignments[tile_y, tile_x, 1])

    mov_y = y + int(s_flow[1])
    mov_x = x + int(s_flow[0])

    s_l1_map = cuda.shared.array(32 * 32, DEFAULT_CUDA_FLOAT_TYPE)
    s_err = cuda.shared.array((16, 16), DEFAULT_CUDA_FLOAT_TYPE) # This one in voluntarly large, we use only (2*search_radius + 1)**2 values
    for shift_y in range(-search_radius, search_radius + 1):
        for shift_x in range(-search_radius, search_radius + 1):
            ### Fancy reduce = sum across threads
            # A presum is made here: a thread manages 4 pixels because the block is 32x32 for a tile size of 64x64
            l1_sum = abs(ref[y, x] - moving[mov_y + shift_y, mov_x + shift_x]) if 0 <= mov_x + shift_x < w and 0 <= mov_y + shift_y < h else 0.0
            l1_sum += abs(ref[y, x + 32] - moving[mov_y + shift_y, mov_x + shift_x + 32]) if 0 <= mov_x + shift_x + 32 < w and 0 <= mov_y + shift_y < h else 0.0
            l1_sum += abs(ref[y + 32, x] - moving[mov_y + shift_y + 32, mov_x + shift_x]) if 0 <= mov_x + shift_x < w and 0 <= mov_y + shift_y + 32 < h else 0.0
            l1_sum += abs(ref[y + 32, x + 32] - moving[mov_y + shift_y + 32, mov_x + shift_x + 32]) if 0 <= mov_x + shift_x + 32 < w and 0 <= mov_y + shift_y + 32 < h else 0.0


            t_per_warp = 32
            w_per_block = (32 * 32) // t_per_warp
            offset = t_per_warp // 2
            # Sum thread among warps first
            while offset > 0:
                l1_sum += cuda.shfl_down_sync(0xffffffff, l1_sum, offset)
                offset //= 2
            if tid % t_per_warp == 0:
                s_l1_map[tid] = l1_sum
            cuda.syncthreads()
            # Then sum warps
            if tid == 0:
                for w_id in range(t_per_warp, w_per_block * t_per_warp, t_per_warp):
                    s_l1_map[0] += s_l1_map[w_id]
            ###########

            
            if tid == 0:
                s_err[shift_y + search_radius, shift_x + search_radius] = s_l1_map[0]


    # Now find the minimum error and corresponding shift
    if tid == 0:
        err = s_err[0, 0]
        for i in range(2*search_radius + 1):
            for j in range(2*search_radius + 1):
                min = s_err[i, j]
                if err < min:
                    min = err
                    min_shift_y = i - search_radius
                    min_shift_x = j - search_radius


    alignments[tile_y, tile_x, 0] = s_flow[0] + min_shift_x
    alignments[tile_y, tile_x, 1] = s_flow[1] + min_shift_y


def extract_flow_patches(frame_tgt, flow, patch_size, radius):
    ny, nx, _ = flow.shape
    p = patch_size
    r = radius
    P_search = 2 * r + p
    frame_tgt = torch.as_tensor(frame_tgt, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
    flow = flow.round().long()

    dx = flow[..., 0]
    dy = flow[..., 1]

    # compute top-left corner of each patch
    top = torch.arange(ny, device=frame_tgt.device)[:, None] * p + dy
    left = torch.arange(nx, device=frame_tgt.device)[None, :] * p + dx

    offsets = torch.arange(P_search, device=frame_tgt.device) - r
    dy_offsets, dx_offsets = torch.meshgrid(offsets, offsets, indexing='ij')

    y_coords = top[:, :, None, None] + dy_offsets[None, None, :, :]
    x_coords = left[:, :, None, None] + dx_offsets[None, None, :, :]

    # clamp to image boundaries
    y_coords = y_coords.clamp(0, frame_tgt.shape[0]-1)
    x_coords = x_coords.clamp(0, frame_tgt.shape[1]-1)

    # flatten for advanced indexing
    y_flat = y_coords.reshape(-1)
    x_flat = x_coords.reshape(-1)

    aligned_patches = frame_tgt[y_flat, x_flat].view(ny, nx, P_search, P_search)
    return aligned_patches