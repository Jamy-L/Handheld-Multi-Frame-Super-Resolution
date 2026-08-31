import time
import math

import numpy as np
from handheld_super_resolution.linalg import bilinear_interpolation
from numba import cuda
import torch
import torch.nn.functional as F
from typing import List
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from .ICA import init_ica, align_lvl_ica
from .block_matching import align_lvl_block_matching_L2, align_lvl_block_matching_L1
from .utils import getTime, clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_CUDA_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE, DEFAULT_THREADS
from .utils_image import cuda_downsample
from .config import Config, AlignmentConfig

SOBEL_Y = torch.as_tensor(np.array([[-1], [0], [1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_X = torch.as_tensor(np.array([[-1,0,1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
SOBEL_Y.requires_grad = False
SOBEL_X.requires_grad = False

def init_alignment(ref_img: DeviceNDArray, config: Config):
    h, w = ref_img.shape

    tile_size = config.alignment.tile_size
    assert isinstance(tile_size, int)
    tile_sizes = config.alignment.tile_sizes

    # if needed, pad image with zeros so that tiles contains all image pixels
    paddingPatchesHeight = (tile_size - h % (tile_size)) * (h % (tile_size) != 0)
    paddingPatchesWidth = (tile_size - w % (tile_size)) * (w % (tile_size) != 0)

    # combine the two to get the total padding
    paddingTop = 0
    paddingBottom = paddingPatchesHeight
    paddingLeft = 0
    paddingRight = paddingPatchesWidth
    
    th_ref_img = torch.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    th_ref_img_padded = F.pad(th_ref_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')



    currentTime, verbose = time.perf_counter(), config.verbose > 2

    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = config.alignment.factors

    pyramid = build_gaussian_pyramid(th_ref_img_padded, factors)

    tiled_fft: List[torch.Tensor] = []
    tiled_pyr: List[torch.Tensor] = []
    gradx_pyramid:  List[DeviceNDArray] = []
    grady_pyramid: List[DeviceNDArray] = []
    hessian_pyramid: List[DeviceNDArray] = []
    for i, lvl in enumerate(pyramid):
        ts = tile_sizes[len(factors) - i - 1]
        gradx, grady, hessian = init_ica(lvl, ts, config)
        tiled = lvl.unfold(0, ts, ts).unfold(1, ts, ts)

        # Pad the crops with 0 to get size 2*R + 1
        r = config.alignment.search_radii[len(factors) - i - 1]
        tiled = torch.nn.functional.pad(tiled, (r, r, r, r), mode='constant', value=0)
        fft = torch.fft.rfft2(tiled, dim=(-2, -1)) # The order of the dim tuple is EXTREMELY important !!! (and undocumented :)))))))
        assert isinstance(fft, torch.Tensor), f"Got invalid fft {fft}"

        tiled_fft.append(fft)
        tiled_pyr.append(tiled)
        gradx_pyramid.append(gradx)
        grady_pyramid.append(grady)
        hessian_pyramid.append(hessian)

    if verbose:
        currentTime = getTime(currentTime, ' --- Create ref pyramid')
    
    return pyramid, tiled_pyr, tiled_fft, gradx_pyramid, grady_pyramid, hessian_pyramid

def build_gaussian_pyramid(image: torch.Tensor, factors: List[int]=[1, 2, 4, 4], kernel='gaussian'):
    pyramid = [cuda_downsample(image, kernel, factors[0])]

    for factor in factors[1:]:
        pyramid.append(cuda_downsample(pyramid[-1], kernel, factor))
    
    pyramid = [lvl.squeeze() for lvl in pyramid]

    return pyramid[::-1]

def align(ref_pyramid: List[torch.Tensor],
          tyled_pyr: List[torch.Tensor],
          ref_tiled_fft: List[torch.Tensor],
          ref_gradx: List[DeviceNDArray],
          ref_grady: List[DeviceNDArray],
          ref_hessian: List[DeviceNDArray],
          img: DeviceNDArray, config: Config):

    th_img = torch.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    
    currentTime, verbose = time.perf_counter(), config.verbose > 2

    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = config.alignment.factors

    moving_pyramid = build_gaussian_pyramid(th_img, factors)

    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' - Create moving pyramid')

    alignments = None
    for l, (ref_lvl, tyled_pyr_lvl, ref_tiled_fft_lvl, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl, moving_lvl) in enumerate(zip(
        ref_pyramid, tyled_pyr, ref_tiled_fft, ref_gradx, ref_grady, ref_hessian, moving_pyramid)):

        list_id = len(ref_pyramid) - l - 1
        # PyTorch uses per-thread default streams; Numba uses the CUDA default stream (presumably)
        # Without this barrier, PyTorch could read 'alignment' before the kernel writes finish.
        cuda.synchronize()
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
    assert alignments is not None
    alignments = cuda.as_cuda_array(alignments) # torch -> numba
    return alignments


def align_lvl(ref_lvl: DeviceNDArray, tyled_pyr_lvl: DeviceNDArray, ref_fft_lvl: DeviceNDArray, ref_gradx_lvl: DeviceNDArray,
              ref_grady_lvl: DeviceNDArray, ref_hessian_lvl: DeviceNDArray, moving_lvl: DeviceNDArray, alignments: torch.Tensor, l: int, config: Config):
    verbose = config.verbose > 2
    currentTime = time.perf_counter()

    metric = config.alignment.block_matching.metrics[l]
    if metric == "L2":
        align_lvl_block_matching_L2(tyled_pyr_lvl, ref_fft_lvl, moving_lvl, alignments, l, config)
    elif metric == "L1":
        align_lvl_block_matching_L1(ref_lvl, moving_lvl, alignments, l, config)
    else:
        raise ValueError("Unknown block matching metric {}".format(metric))

    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' -- Block matching level {}'.format(l))

    align_lvl_ica(ref_lvl, ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
                  moving_lvl, alignments, l, config)
    
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' -- ICA level {}'.format(l))


def upscale_lvl(alignments: torch.Tensor, npatchs: List[int], l: int, config: Config):
    # Upscale (and scale...) the flow for the next pyramid level
    new_tile_size = config.alignment.tile_sizes[l]
    prev_tile_size = config.alignment.tile_sizes[l+1]
    upsampling_factor = config.alignment.factors[l+1]

    repeat_factor = upsampling_factor // (new_tile_size // prev_tile_size)

    upsampled_alignments= F.interpolate(
        alignments.permute(2, 0, 1)[None], scale_factor=repeat_factor,
        mode=config.alignment.flow_upscale_mode)[0].permute(1, 2, 0)
    assert isinstance(upsampled_alignments, torch.Tensor), f"Invalid upsample: {upsampled_alignments}"
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
    
