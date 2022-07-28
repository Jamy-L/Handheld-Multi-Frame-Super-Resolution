# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:41:14 2022

@author: jamyl
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import matplotlib
from time import time

from torch import Tensor
from typing import Tuple, List, Optional, Dict



class Layer():
    def __init__(self, index: int, array,
                 parameters: Dict[str, List[int]],
                 downscale_factor: int,
                 prev_alignment: Tensor = torch.zeros([2, 1, 1])):
        self.index = index
        self.array = array
        self.prev_alignment = prev_alignment

        _, self.layer_h, self. layer_w = array.shape
        self.tile_h, self.tile_w = parameters['tile shape']
        self.downscale_factor = downscale_factor
        self.search_region = parameters['search region']

        self.n_tiles_y = self.layer_h // (self.tile_h // 2) - 1
        self.n_tiles_x = self.layer_w // (self.tile_w // 2) - 1

        # dummy values, init as tensors. tile_estimated enables to know
        # if they really have been estimated
        self.tile_idx = torch.zeros(1)
        self.tile_idy = torch.zeros(1)

        self.tile_estimated = False
        self.alignment = torch.zeros(1)

    def upscale_prev_alignment(self):
        """
        When layers in an image pyramid are iteratively compared,
        the absolute pixel distances in each layer represent different
        relative distances. This function interpolates an optical flow
        from one resolution to another, taking care to scale the values.
        """
        alignment = self.prev_alignment[None].float(
        )  # [1, 2, n_tiles_y, n_tiles_x]
        alignment = F.interpolate(alignment, size=(
            self.n_tiles_y, self.n_tiles_x), mode='nearest')
        alignment *= self.downscale_factor  # [1, 2, n_tiles_y, n_tiles_x]
        alignment = (alignment[0]).to(torch.int16)  # [2, n_tiles_y, n_tiles_x]
        self.prev_alignment = alignment

    def compute_tile_indices(self):
        """
        Computes the indices of tiles along the layer.
        """
        # compute x indices
        x_min = torch.linspace(0, self.layer_w-self.tile_w,
                               self.n_tiles_x,
                               dtype=torch.int16)  # [n_tiles_x]
        dx = torch.arange(0, self.tile_w)  # [tile_w]
        x = x_min[:, None] + dx[None, :]  # [n_tiles_x, tile_w]
        # [n_tiles_y, n_tiles_x, tile_h, tile_w]
        x = x[None, :, None, :].repeat(self.n_tiles_y, 1, self.tile_h, 1)

        # compute y indices
        y_min = torch.linspace(0, self.layer_h-self.tile_h, self.n_tiles_y,
                               dtype=torch.int16)  # [n_tiles_y]
        dy = torch.arange(0, self.tile_h)  # [tile_h]
        y = y_min[:, None] + dy[None, :]  # [n_tiles_y, tile_h]
        # [n_tiles_y, n_tiles_x, tile_h, tile_w]
        y = y[:, None, :, None].repeat(1, self.n_tiles_x, 1, self.tile_w)

        # (2 x [n_tiles_y, n_tiles_x, tile_h, tile_w])
        self.tile_idx, self.tile_idy = x, y
        self.tile_estimated = True

    def shift_tile_indices(self):
        """
        Given a tensor of tile indices, shift these indices by
        [search_dist_min, search_dist_max] in both x- and y-directions.
        This creates (search_dist_max - search_dist_min + 1)^2
        tile "displacements". All the displacements are returned
        along the first (batch) dimension. To recover the displacement
        from the batch dimension, use the following conversion:
        >>> dy = idx // n_pos + search_dist_min
        >>> dx = idx % n_pos + search_dist_min
        """
        search_dist_min, search_dist_max = self.search_region[0], self.search_region[1]
        # create a tensor of displacements
        ds = torch.arange(search_dist_min, search_dist_max+1)
        n_pos = len(ds)

        # compute x indices
        # [n_tiles_y, n_tiles_x, tile_h, tile_w, 1, n_pos]
        x = self.tile_idx[:, :, :, :, None, None] + \
            ds[None, None, None, None, None, :]
        # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos, n_pos]
        x = x.repeat(1, 1, 1, 1, n_pos, 1)
        # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos*n_pos]
        x = x.flatten(-2)
        # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
        x = x.permute(4, 0, 1, 2, 3)

        # compute y indices
        # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos, 1]
        y = self.tile_idy[:, :, :, :, None, None] + \
            ds[None, None, None, None, :, None]
        # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos, n_pos]
        y = y.repeat(1, 1, 1, 1, 1, n_pos)
        # [n_tiles_y, n_tiles_x, tile_h, tile_w, n_pos*n_pos]
        y = y.flatten(-2)
        # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
        y = y.permute(4, 0, 1, 2, 3)

        # (2 x [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w])
        self.tile_idx, self.tile_idy = x, y

    def clamp_tile_id(self):
        """
        Clamp indices to layer shape.
        """
        # use inplace operations to reduce momory usage
        self.tile_idx = self.tile_idx.clamp_(0, self.layer_w-1)
        self.tile_idy = self.tile_idy.clamp_(0, self.layer_h-1)


# @torch.jit.script
def l1_brute_force_align(comp_layer: Layer, ref_layer: Layer) -> Tensor:

    search_dist_min, search_dist_max = comp_layer.search_region
    n_pos = search_dist_max - search_dist_min + 1

    # gather tiles from the reference layer
    # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    if not ref_layer.tile_estimated:
        ref_layer.compute_tile_indices()
    # [1, n_tiles_y, n_tiles_x, tile_h, tile_w]
    ref_tiles = ref_layer.array[:, ref_layer.tile_idy, ref_layer.tile_idx]

    # compute coordinates for comparison tiles
    # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    if not comp_layer.tile_estimated:
        comp_layer.compute_tile_indices()
    if comp_layer.prev_alignment is not None:
        comp_layer.upscale_prev_alignment()

    comp_layer.tile_idx += comp_layer.prev_alignment[0, :, :, None, None]
    comp_layer.tile_idy += comp_layer.prev_alignment[1, :, :, None, None]
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_layer.shift_tile_indices()

    # check if each comparison tile is fully within the layer dimensions
    tile_is_inside_layer = torch.ones(
        [n_pos*n_pos, comp_layer.n_tiles_y, comp_layer.n_tiles_x], dtype=torch.bool)
    tile_is_inside_layer &= comp_layer.tile_idx[:, :, :,  0,  0] >= 0
    tile_is_inside_layer &= comp_layer.tile_idx[:,
                                                :, :,  0, -1] < comp_layer.layer_w
    tile_is_inside_layer &= comp_layer.tile_idy[:, :, :,  0,  0] >= 0
    tile_is_inside_layer &= comp_layer.tile_idy[:,
                                                :, :, -1,  0] < comp_layer.layer_h

    # gather tiles from the comparison layer
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_layer.clamp_tile_id()
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_tiles = comp_layer.array[0, comp_layer.tile_idy, comp_layer.tile_idx]

    # compute the difference between the comparison and reference tiles
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    diff = comp_tiles - ref_tiles
    # [n_pos*n_pos, n_tiles_y, n_tiles_x]
    diff = diff.abs().sum(dim=[-2, -1])

    # set the difference value for tiles outside of the frame to infinity
    diff[~tile_is_inside_layer] = float('inf')

    # find which shift (dx, dy) between the reference and comparison tiles yields the lowest loss
    # - 'torch.div' is used here instead of '//' to avoid a 'UserWarning' in PyTorch
    #   however, 'torch.div' and '//' are equivalent for non-negative values
    #   https://github.com/pytorch/pytorch/issues/43874
    argmin = diff.argmin(0)  # [n_tiles_y, n_tiles_x]
    dy = torch.div(argmin, n_pos, rounding_mode='floor') + \
        search_dist_min  # [n_tiles_y, n_tiles_x]
    dx = argmin % n_pos + search_dist_min  # [n_tiles_y, n_tiles_x]

    # save the current alignment
    alignment = torch.stack([dx, dy], 0)  # [2, n_tiles_y, n_tiles_x]

    # combine the current alignment with the previous alignment
    alignment += comp_layer.prev_alignment

    return alignment


# @torch.jit.script
def l2_fast_align(comp_layer: Layer, ref_layer: Layer) -> Tensor:
    # TODO D2 is fast computed, but the minimisation remains slow
    search_dist_min, search_dist_max = comp_layer.search_region
    n_pos = search_dist_max - search_dist_min + 1

    # gather tiles from the reference layer
    # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    if not ref_layer.tile_estimated:
        ref_layer.compute_tile_indices()
    # [1, n_tiles_y, n_tiles_x, tile_h, tile_w]
    ref_tiles = ref_layer.array[0, ref_layer.tile_idy, ref_layer.tile_idx]

    # compute coordinates for comparison tiles
    # [n_tiles_y, n_tiles_x, tile_h, tile_w]
    if not comp_layer.tile_estimated:
        comp_layer.compute_tile_indices()

    comp_layer.upscale_prev_alignment()

    comp_layer.tile_idx += comp_layer.prev_alignment[0, :, :, None, None]
    comp_layer.tile_idy += comp_layer.prev_alignment[1, :, :, None, None]
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_layer.shift_tile_indices()

    # check if each comparison tile is fully within the layer dimensions
    tile_is_inside_layer = torch.ones(
        [n_pos*n_pos, comp_layer.n_tiles_y, comp_layer.n_tiles_x], dtype=torch.bool)
    tile_is_inside_layer &= comp_layer.tile_idx[:, :, :,  0,  0] >= 0
    tile_is_inside_layer &= comp_layer.tile_idx[:,
                                                :, :,  0, -1] < comp_layer.layer_w
    tile_is_inside_layer &= comp_layer.tile_idy[:, :, :,  0,  0] >= 0
    tile_is_inside_layer &= comp_layer.tile_idy[:,
                                                :, :, -1,  0] < comp_layer.layer_h

    # gather tiles from the comparison layer
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_layer.clamp_tile_id()
    # [n_pos*n_pos, n_tiles_y, n_tiles_x, tile_h, tile_w]
    comp_tiles = comp_layer.array[0, comp_layer.tile_idy, comp_layer.tile_idx]

    # [n_pos*n_pos, n_tiles_y, n_tiles_x]
    comp_norms = torch.linalg.matrix_norm(comp_tiles)
    # [n_tiles_y, n_tiles_x]
    ref_norms = torch.linalg.matrix_norm(ref_tiles)

    # The following tile family correspond to tiles shifted only by the result
    # of the last step.
    comp_tiles_origin = comp_tiles[n_pos*n_pos//2,
                                   :, :, :, :]
    fft_ref = torch.fft.fft2(ref_tiles, dim=(- 2, - 1))
    fft_comp = torch.fft.fft2(comp_tiles_origin, dim=(- 2, - 1))
    conv = torch.fft.ifft2(
        fft_ref.conj().transpose(2, 3) * fft_comp, s=(n_pos, n_pos),
        dim=(- 2, - 1),
        norm="ortho"
    )
    # from [n_tiles_y, n_tyles_x, n_pos, n_pos]
    # to [n_pos*n_pos, n_tiles_y, n_tyles_x]
    conv = conv.permute(2, 3, 0, 1)
    conv = conv.reshape(n_pos*n_pos, comp_layer.n_tiles_y,
                        comp_layer.n_tiles_x)

    D2 = ref_norms*ref_norms + comp_norms*comp_norms - 2*torch.real(conv)
    # find which shift (dx, dy) between the reference and comparison tiles yields the lowest loss
    # - 'torch.div' is used here instead of '//' to avoid a 'UserWarning' in PyTorch
    #   however, 'torch.div' and '//' are equivalent for non-negative values
    #   https://github.com/pytorch/pytorch/issues/43874
    argmin = D2.argmin(0)  # [n_tiles_y, n_tiles_x]
    dy = torch.div(argmin, n_pos, rounding_mode='floor') + \
        search_dist_min  # [n_tiles_y, n_tiles_x]
    dx = argmin % n_pos + search_dist_min  # [n_tiles_y, n_tiles_x]

    # save the current alignment
    alignment = torch.stack([dx, dy], 0)  # [2, n_tiles_y, n_tiles_x]

    # combine the current alignment with the previous alignment
    alignment += comp_layer.prev_alignment

    return alignment


#%%
parameters = {'tile shape':[2,2],
              'search region':[-4,4]
              }
downscale_factor=1
prev_alignment = torch.zeros([2, 1, 1])


ref_array = torch.Tensor([[0, 0, 1, 1, 2, 2],
                          [0, 0, 1, 1, 2, 2],
                          [3, 3, 4, 4, 5, 5],
                          [3, 3, 4, 4, 5, 5],
                          [6, 6, 7, 7, 8, 8],
                          [6, 6, 7, 7, 8, 8]])[None]

comp_array = ref_array.roll((2,2), (0,1))
ref_layer = Layer(index = 0,
                  array = ref_array,
                  parameters= parameters,
                  downscale_factor=downscale_factor,
                  prev_alignment=prev_alignment)

comp_layer = Layer(index = 1,
                  array = comp_array,
                  parameters= parameters,
                  downscale_factor=downscale_factor,
                  prev_alignment=prev_alignment)

plt.figure('ref')
plt.imshow(np.array(ref_array[0]))
plt.title("ref")
plt.figure('comp')
plt.imshow(np.array(comp_array[0]))
plt.title("comp")

print(l1_brute_force_align(comp_layer, ref_layer))



















