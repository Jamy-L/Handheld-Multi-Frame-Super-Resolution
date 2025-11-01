import time
import math

import numpy as np
from handheld_super_resolution.linalg import bilinear_interpolation
from numba import cuda
import torch
import torch.nn.functional as F

from .utils import getTime, clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS
from .utils_image import cuda_downsample
from .linalg import solve_2x2

SOBEL_Y = torch.as_tensor(np.array([[-1], [0], [1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_X = torch.as_tensor(np.array([[-1,0,1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_Y.requires_grad = False
SOBEL_X.requires_grad = False

BOX_FILTER_8 = torch.as_tensor(np.ones((1,1,8,8)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_8.requires_grad = False
BOX_FILTER_16 = torch.as_tensor(np.ones((1,1,16,16)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_16.requires_grad = False
BOX_FILTER_32 = torch.as_tensor(np.ones((1,1,32,32)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_32.requires_grad = False
BOX_FILTER_64 = torch.as_tensor(np.ones((1,1,64,64)), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
BOX_FILTER_64.requires_grad = False

def init_alignment(ref_img, config):
    h, w = ref_img.shape

    tileSize = config.block_matching.tuning.tile_size
    tileSizes = config.block_matching.tuning.tile_sizes

    # if needed, pad image with zeros so that tiles contains all image pixels
    paddingPatchesHeight = (tileSize - h % (tileSize)) * (h % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - w % (tileSize)) * (w % (tileSize) != 0)

    # combine the two to get the total padding
    paddingTop = 0
    paddingBottom = paddingPatchesHeight
    paddingLeft = 0
    paddingRight = paddingPatchesWidth
    
    th_ref_img = torch.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    th_ref_img_padded = F.pad(th_ref_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')



    currentTime, verbose = time.perf_counter(), config.verbose > 2

    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = config.block_matching.tuning.factors

    pyramid = build_gaussian_pyramid(th_ref_img_padded, factors)

    tiled_fft = []
    tiled_pyr = []
    gradx_pyramid = []
    grady_pyramid = []
    hessian_pyramid = []
    for i, lvl in enumerate(pyramid):
        gradx, grady, hessian = init_ica(lvl, config)

        ts = tileSizes[len(factors) - i - 1]
        tiled = lvl.unfold(0, ts, ts).unfold(1, ts, ts)

        # Pad the crops with 0 to get size 2*R + 1
        r = config.block_matching.tuning.search_radii[len(factors) - i - 1]
        tiled = torch.nn.functional.pad(tiled, (r, r, r, r), mode='constant', value=0)
        fft = torch.fft.rfft2(tiled, dim=(-2, -1)) # The order of the dim tuple is EXTREMELY important !!! (and undocumented :)))))))

        tiled_fft.append(fft)
        tiled_pyr.append(tiled)
        gradx_pyramid.append(gradx)
        grady_pyramid.append(grady)
        hessian_pyramid.append(hessian)

    if verbose:
        currentTime = getTime(currentTime, ' --- Create ref pyramid')
    
    return pyramid, tiled_pyr, tiled_fft, gradx_pyramid, grady_pyramid, hessian_pyramid

def build_gaussian_pyramid(image, factors=[1, 2, 4, 4], kernel='gaussian'):
    pyramid = [cuda_downsample(image, kernel, factors[0])]

    for factor in factors[1:]:
        pyramid.append(cuda_downsample(pyramid[-1], kernel, factor))
    
    pyramid = [lvl.squeeze() for lvl in pyramid]

    return pyramid[::-1]

def init_ica(image, config):
    imsize_y, imsize_x = image.shape
    tile_size = config.block_matching.tuning.tile_size
    n_patch_y = math.ceil(imsize_y / tile_size)
    n_patch_x = math.ceil(imsize_x / tile_size)

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

def align(ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian,
          img, config):

    th_img = torch.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    currentTime, verbose = time.perf_counter(), config.verbose > 2

    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = config.block_matching.tuning.factors

    moving_pyramid = build_gaussian_pyramid(th_img, factors)

    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' - Create moving pyramid')

    alignments = None
    for l, (ref_lvl, tyled_pyr_lvl, ref_tiled_fft_lvl, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl, moving_lvl) in enumerate(zip(
        ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian, moving_pyramid)):

        list_id = len(ref_pyramid) - l - 1
        if alignments is None:
            alignments = torch.zeros((*ref_tiled_fft_lvl.shape[:2], 2), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
        else:
            alignments = upscale_lvl(alignments, ref_tiled_fft_lvl.shape[:2], list_id, config) # Juste a re-tiling and scaling

        with torch.no_grad():
            align_lvl(
                ref_lvl, tyled_pyr_lvl, ref_tiled_fft_lvl, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
                moving_lvl, alignments, l=list_id, config=config)
            
        if verbose:
            cuda.synchronize()
            currentTime = getTime(currentTime, ' - Align pyramid')

    return alignments
    


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


def align_lvl(ref_lvl, tyled_pyr_lvl, ref_fft_lvl, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
            moving_lvl, alignments, l, config):
    verbose = config.verbose > 2
    currentTime = time.perf_counter()

    metric = config.block_matching.tuning.metrics[l]
    if metric == "L2":
        align_lvl_block_matching_L2(tyled_pyr_lvl, ref_fft_lvl, moving_lvl, alignments, l, config)
    elif metric == "L1":
        align_lvl_block_matching_L1(ref_lvl, moving_lvl, alignments, l, config)
    else:
        raise ValueError("Unknown block matching metric {}".format(config.block_matching.metric))

    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' -- Block matching level {}'.format(l))

    align_lvl_ica(ref_lvl, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
                  moving_lvl, alignments, l, config)
    
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' -- ICA level {}'.format(l))
    

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

    # Old way a thread per patch
    # threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    # blockspergrid_x = math.ceil(w / threadsperblock[1])
    # blockspergrid_y = math.ceil(h / threadsperblock[0])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)

    # cuda_L1_local_search[blockspergrid, threadsperblock](
    #     ref_lvl, moving_lvl, tile_size, search_radius, alignments)
    
    # New way, 1 thread per pixel
    threadsperblock = (tile_size, tile_size)
    blockspergrid_x = math.ceil(nx)
    blockspergrid_y = math.ceil(ny)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda_L1_local_search_new[blockspergrid, threadsperblock](
        ref_lvl, moving_lvl, tile_size, search_radius, alignments)

@cuda.jit
def cuda_L1_local_search_new(ref, moving, tile_size, search_radius, alignments):
    h, w = moving.shape
    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tile_x = cuda.blockIdx.x
    tile_y = cuda.blockIdx.y
    tid = ty * tile_size + tx
    
    local_flow = cuda.shared.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        local_flow[0] = round(alignments[tile_y, tile_x, 0])
        local_flow[1] = round(alignments[tile_y, tile_x, 1])

    # Load ref patch into shared memory
    s_ref = cuda.shared.array((16, 16), DEFAULT_CUDA_FLOAT_TYPE)
    s_ref[ty, tx] = ref[y, x] if (0 <= x < w and 0 <= y < h) else 0.0
    cuda.syncthreads()

    mov_y = y + int(local_flow[1]) - search_radius
    mov_x = x + int(local_flow[0]) - search_radius

    s_mov = cuda.shared.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    s_mov[ty, tx] = moving[mov_y, mov_x] if 0 <= mov_x < w and 0 <= mov_y < h else 0.0

    # Load the remaining pixels
    if tx < 2 * search_radius:
        s_mov[ty, tx + tile_size] = moving[mov_y, mov_x + tile_size] if 0 <= mov_x + tile_size < w and 0 <= mov_y < h else 0.0
    if ty < 2 * search_radius:
        s_mov[ty + tile_size, tx] = moving[mov_y + tile_size, mov_x] if 0 <= mov_x < w and 0 <= mov_y + tile_size < h else 0.0
    if tx < 2 * search_radius and ty < 2 * search_radius:
        s_mov[ty + tile_size, tx + tile_size] = moving[mov_y + tile_size, mov_x + tile_size] if 0 <= mov_x + tile_size < w and 0 <= mov_y + tile_size < h else 0.0
    cuda.syncthreads()

    s_l1_map = cuda.shared.array(16 * 16, DEFAULT_CUDA_FLOAT_TYPE)
    s_err = cuda.shared.array((16, 16), DEFAULT_CUDA_FLOAT_TYPE)
    for shift_y in range(-search_radius, search_radius + 1):
        for shift_x in range(-search_radius, search_radius + 1):
            ### Fancy reduce = sum accros threads
            l1_sum = abs(s_ref[ty, tx] - s_mov[ty + shift_y + search_radius, tx + shift_x + search_radius])
            t_per_warp = 32
            w_per_block = (tile_size * tile_size) // t_per_warp
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


    alignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
    alignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y

@cuda.jit
def cuda_L1_local_search(ref, moving, tile_size, search_radius, alignments):
    n_patchs_y, n_patchs_x, _ = alignments.shape
    h, w = moving.shape
    tile_x, tile_y = cuda.grid(2)
    if not(0 <= tile_y < n_patchs_y and
           0 <= tile_x < n_patchs_x):
        return
    
    local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_flow[0] = alignments[tile_y, tile_x, 0]
    local_flow[1] = alignments[tile_y, tile_x, 1]

    # position of the pixel in the top left corner of the patch
    patch_pos_x = tile_x * tile_size
    patch_pos_y = tile_y * tile_size

    # this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tile_size):
        for j in range(tile_size):
            idx = patch_pos_x + j
            idy = patch_pos_y + i
            local_ref[i, j] = ref[idy, idx]

    min_dist = +1/0 #init as infty
    min_shift_y = 0
    min_shift_x = 0
    # window search
    for search_shift_y in range(-search_radius, search_radius + 1):
        for search_shift_x in range(-search_radius, search_radius + 1):
            # computing dist
            dist = 0
            for i in range(tile_size):
                for j in range(tile_size):
                    new_idx = patch_pos_x + j + int(local_flow[0]) + search_shift_x
                    new_idy = patch_pos_y + i + int(local_flow[1]) + search_shift_y
                    
                    if (0 <= new_idx < w and
                        0 <= new_idy < h):
                        dist += abs(local_ref[i, j] - moving[new_idy, new_idx])
                    else:
                        dist = +1/0
                    
                    
            if dist < min_dist:
                min_dist = dist
                min_shift_y = search_shift_y
                min_shift_x = search_shift_x

    alignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
    alignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y



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

def align_lvl_ica(ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
                  moving_lvl, alignment, l, config):
    verbose_3 = config.verbose >= 3
    tile_size = config.block_matching.tuning.tile_size

    n_patch_y, n_patch_x, _ = alignment.shape
    h, w = moving_lvl.shape
    
    # # Old way, 1 thread/patch
    # threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    
    # blockspergrid_x = math.ceil(n_patch_x / threadsperblock[1])
    # blockspergrid_y = math.ceil(n_patch_y / threadsperblock[0])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # cuda_ica[blockspergrid, threadsperblock](
    #     ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
    #     moving_lvl, alignment, tile_size, config.ica.tuning.n_iter)   

    # New way, 1 thread/pixel
    threadsperblock = (tile_size, tile_size)
    blockspergrid_x = math.ceil(n_patch_x)
    blockspergrid_y = math.ceil(n_patch_y)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    if tile_size == 8:
        cuda_kernel = ica_kernel_8
    elif tile_size == 16:
        cuda_kernel = ica_kernel_16
    elif tile_size == 32:
        cuda_kernel = ica_kernel_32
    elif tile_size == 64:
        cuda_kernel = ica_kernel_64
    else:
        raise NotImplementedError("ICA kernel for tile size {} not implemented".format(tile_size))
    cuda_kernel[blockspergrid, threadsperblock](
        ref_img, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
        moving_lvl, alignment, config.ica.tuning.n_iter)


@cuda.jit
def ica_kernel_16(ref_img, gradx, grady, hessian, moving, alignment, niter):
    tile_size = 16
    imsize_y, imsize_x = moving.shape
    n_patchs_y, n_patchs_x, _ = alignment.shape
    x, y  = cuda.grid(2)
    patch_idx = cuda.blockIdx.x
    patch_idy = cuda.blockIdx.y
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tid = ty * tile_size + tx
    # NOTE: There can be patches partialy out of frame. In this case, some x,y are out of image bounds.
    # I strip them out of the sums by setting relevant quantities to 0.

    if not(0 <= patch_idy < n_patchs_y and
        0 <= patch_idx < n_patchs_x):
        return
    
    is_inbound = x < imsize_x and y < imsize_y
    
    A00 = hessian[patch_idy, patch_idx, 0, 0]
    A01 = hessian[patch_idy, patch_idx, 0, 1]
    A10 = hessian[patch_idy, patch_idx, 1, 0]
    A11 = hessian[patch_idy, patch_idx, 1, 1]

    det = A00 * A11 - A01 * A10
    if abs(det) < 1e-10: # system is Not solvable
        return  # 1 Hessian per block, so all threads exit
    det_inv = 1.0 / det

    s_alignment = cuda.shared.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    if tid == 0:
        s_alignment[0] = alignment[patch_idy, patch_idx, 0]
        s_alignment[1] = alignment[patch_idy, patch_idx, 1]

    l_grad = cuda.local.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    l_grad[0] = gradx[y, x] if is_inbound else 0.0
    l_grad[1] = grady[y, x] if is_inbound else 0.0

    s_B_0 = cuda.shared.array((16 * 16), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    s_B_1 = cuda.shared.array((16 * 16), dtype = DEFAULT_CUDA_FLOAT_TYPE)

    ref_c = ref_img[y, x] if is_inbound else 0.0

    for _ in range(niter):
        cuda.syncthreads()
        # Warp I with W(x; p) to compute I(W(x; p))

        ## bilinear interpolation at new_x, new_y
        floor_x = x + int(s_alignment[0])
        floor_y = y + int(s_alignment[1])
        frac_x, _ = math.modf(s_alignment[0])
        frac_y, _ = math.modf(s_alignment[1])
        # Note: in theory frac_x, floor_x = math.modf(x + alignment[0]) in 1 shot. But it is surprisingly faster to compute it from s_alignment this way 

        floor_x = clamp(floor_x, 0, imsize_x - 1)
        floor_y = clamp(floor_y, 0, imsize_y - 1)

        ceil_x = clamp(floor_x + 1, 0, imsize_x - 1)
        ceil_y = clamp(floor_y + 1, 0, imsize_y - 1)

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
        s_B_0[tid] = -l_grad[0] * gradt
        s_B_1[tid] = -l_grad[1] * gradt
        N = tile_size * tile_size // 2
        while N > 0:
            cuda.syncthreads()
            if tid < N:
                s_B_0[tid] += s_B_0[tid + N]
                s_B_1[tid] += s_B_1[tid + N]
            N = N // 2
        #############

        if tid == 0:
            B0 = s_B_0[0]
            B1 = s_B_1[0]

            # solve Ax = B
            s_alignment[0] += det_inv * (A11 * B0 - A01 * B1)
            s_alignment[1] += det_inv * (-A10 * B0 + A00 * B1)

    if tid == 0:
        alignment[patch_idy, patch_idx, 0] = s_alignment[0]
        alignment[patch_idy, patch_idx, 1] = s_alignment[1]

@cuda.jit
def cuda_ica(ref_img, gradx, grady, hessian, moving, alignment, tile_size, niter):
    imsize_y, imsize_x = moving.shape
    n_patchs_y, n_patchs_x, _ = alignment.shape
    patch_idx, patch_idy = cuda.grid(2)
    
    if not(0 <= patch_idy < n_patchs_y and
           0 <= patch_idx < n_patchs_x):
        return
    
    patch_pos_x = tile_size * patch_idx
    patch_pos_y = tile_size * patch_idy
    
    A = cuda.local.array((2,2), dtype = DEFAULT_CUDA_FLOAT_TYPE)
    A[0, 0] = hessian[patch_idy, patch_idx, 0, 0]
    A[0, 1] = hessian[patch_idy, patch_idx, 0, 1]
    A[1, 0] = hessian[patch_idy, patch_idx, 1, 0]
    A[1, 1] = hessian[patch_idy, patch_idx, 1, 1]
    
    if abs(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]) < 1e-10: # system is Not solvable
        return 
    
    B = cuda.local.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    local_alignment = cuda.local.array(2, dtype = DEFAULT_CUDA_FLOAT_TYPE)
    local_alignment[0] = alignment[patch_idy, patch_idx, 0]
    local_alignment[1] = alignment[patch_idy, patch_idx, 1]
    
    buffer_val = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE) # y, x

    for i in range(niter):
        B[0] = 0
        B[1] = 0
        for i in range(tile_size):
            for j in range(tile_size):
                pixel_global_idx = patch_pos_x + j # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
                pixel_global_idy = patch_pos_y + i
                
                if not (0 <= pixel_global_idx < imsize_x and 
                        0 <= pixel_global_idy < imsize_y):
                    continue
                
                local_gradx = gradx[pixel_global_idy, pixel_global_idx]
                local_grady = grady[pixel_global_idy, pixel_global_idx]

                # Warp I with W(x; p) to compute I(W(x; p))
                new_idx = local_alignment[0] + pixel_global_idx
                new_idy = local_alignment[1] + pixel_global_idy 
    
                if not (0 <= new_idx < imsize_x - 1 and
                        0 <= new_idy < imsize_y - 1): # -1 for bicubic interpolation
                    continue
                
                # bilinear interpolation
                normalised_pos_x, floor_x = math.modf(new_idx) # https://www.rollpie.com/post/252
                normalised_pos_y, floor_y = math.modf(new_idy) # separating floor and floating part
                floor_x = int(floor_x)
                floor_y = int(floor_y)
                
                ceil_x = floor_x + 1
                ceil_y = floor_y + 1
                pos[0] = normalised_pos_y
                pos[1] = normalised_pos_x

                buffer_val[0, 0] = moving[floor_y, floor_x]
                buffer_val[0, 1] = moving[floor_y, ceil_x]
                buffer_val[1, 0] = moving[ceil_y, floor_x]
                buffer_val[1, 1] = moving[ceil_y, ceil_x]

                comp_val = bilinear_interpolation(buffer_val, pos)
                
                gradt = comp_val - ref_img[pixel_global_idy, pixel_global_idx]
                
                
                B[0] += -local_gradx*gradt
                B[1] += -local_grady*gradt
        
    
        alignment_step = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    
        # solvability is ensured by design
        solve_2x2(A, B, alignment_step)
        local_alignment[0] += alignment_step[0]
        local_alignment[1] += alignment_step[1]
    
    alignment[patch_idy, patch_idx, 0] = local_alignment[0]
    alignment[patch_idy, patch_idx, 1] = local_alignment[1]

def upscale_lvl(alignments, npatchs, l, config):
    # Upscale (and scale...) the flow for the next pyramid level
    new_tile_size = config.block_matching.tuning.tile_sizes[l]
    prev_tile_size = config.block_matching.tuning.tile_sizes[l+1]
    upsampling_factor = config.block_matching.tuning.factors[l+1]

    repeat_factor = upsampling_factor // (new_tile_size // prev_tile_size)

    upsampled_alignments = torch.repeat_interleave(
        torch.repeat_interleave(
            alignments, repeat_factor, dim=0), repeat_factor, dim=1)
    upsampled_alignments *= upsampling_factor

    # Add a potential tile with 0 flow on the bottom or the right
    if upsampled_alignments.shape[0] < npatchs[0] or upsampled_alignments.shape[1] < npatchs[1]:
        upsampled_alignments = F.pad(
            upsampled_alignments,
            (0, 0,
             0, npatchs[1] - upsampled_alignments.shape[1],
             0, npatchs[0] - upsampled_alignments.shape[0]),
            mode='constant', value=0)

    return upsampled_alignments
    
