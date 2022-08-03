# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:00:21 2022

##@author: jamyl
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from time import time

from torch import Tensor
from typing import Tuple, List, Optional, Dict

# %%


def timer(func):
    def run(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        print("execution time : ", time()-t1)
        return result
    return run


@torch.jit.script
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


@torch.jit.script
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


@torch.jit.script
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


@torch.jit.script
class Image():
    def __init__(self, array):
        self.rgb_array = array
        self.bnw_array = torch.mean(array, dim=0, keepdim=True)
        self.shape = (array.shape[1], array.shape[2])
        self.pyramid: List[Layer] = []

    def build_pyramid(self,
                      downscale_factors: List[int],
                      parameters: Dict[str, List[List[int]]]):

        layer_array = self.bnw_array
        for layer_index, downscale_factor in enumerate(downscale_factors):
            layer_array = F.avg_pool2d(layer_array, downscale_factor)
            layer_parameters = {
                "tile shape": parameters['tile shapes'][layer_index],
                "search region": parameters['search regions'][layer_index]

            }
            layer = Layer(layer_index,
                          layer_array,
                          layer_parameters,
                          downscale_factors[layer_index],
                          torch.zeros([2, 1, 1]))
            self.pyramid.append(layer)


def plot_alignment(compared_image: Image, ref_image: Image):
    def complex_array_to_rgb(X, theme='dark', rmax=None):
        '''Takes an array of complex number and converts it to an array of [r, g, b],
        where phase gives hue and saturaton/value are given by the absolute value.
        Especially for use with imshow for complex plots.'''
        absmax = rmax or np.abs(X).max()
        Y = np.zeros(X.shape + (3,), dtype='float')
        Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
        if theme == 'light':
            Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
            Y[..., 2] = 1
        elif theme == 'dark':
            Y[..., 1] = 1
            Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
        Y = matplotlib.colors.hsv_to_rgb(Y)
        return Y

    layers = compared_image.pyramid
    n_layers = len(layers)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(compared_image.bnw_array[0, :, :]).astype(np.int32),
               cmap=plt.get_cmap('gray'))
    plt.title("compared")

    plt.subplot(1, 2, 2)
    plt.imshow(np.array(ref_image.bnw_array[0, :, :]).astype(np.int32),
               cmap=plt.get_cmap('gray'))
    plt.title("reference")

    plt.figure()
    for i, layer in enumerate(layers):
        plt.subplot(1, n_layers, i+1)
        plt.imshow(complex_array_to_rgb(
            layer.alignment[0, :, :] + 1j*layer.alignment[1, :, :],
            theme="light", rmax=5))

    # legend
    X = np.linspace(-50, 50, 100)
    X = np.stack((X,)*100, axis=1)
    Y = np.rot90(X)
    Z = (X+1j*Y)/10
    plt.figure("legend")
    plt.imshow(complex_array_to_rgb(
        Z,
        theme="light", rmax=5))


@torch.jit.script
class ImageBurst():
    def __init__(self, reference_image: Image,
                 images: List[Image],
                 downscale_factors: List[int],
                 parameters: Dict[str, List[List[int]]]):
        self.reference_image = reference_image
        self.images = images
        self.downscale_factors = downscale_factors
        self.parameters = parameters
        self.burst_size = len(images)
        self.shape = (images[0].shape[0], images[0].shape[1])

    def build_pyramids(self):
        self.reference_image.build_pyramid(
            self.downscale_factors,
            self.parameters)
        for image in self.images:
            image.build_pyramid(self.downscale_factors,
                                self.parameters)

    def estimate_images_alignments(self):
        for image in self.images:
            self.estimate_image_alignment(image)

    def estimate_image_alignment(self, image: Image):
        prev_alignment: Tensor = torch.zeros([2, 1, 1])
        for layer_index in torch.flip(torch.arange(len(image.pyramid)), [0]):
            image.pyramid[layer_index].prev_alignment = prev_alignment
            alignment = l1_brute_force_align(image.pyramid[layer_index],
                                      self.reference_image.pyramid[layer_index])
            prev_alignment = alignment
            image.pyramid[layer_index].alignment = alignment


# %% importing data
frames = []
cap = cv2.VideoCapture('Book_case1.avi')
ret, frame = cap.read()
while ret:
    frame = Tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = frame.permute(2, 0, 1)

    frames.append(Image(frame))
    ret, frame = cap.read()


downscale_factors = [1, 2, 2, 4]
# first level : min L1 by bruteforce
# 3,2,4 level : fast L2
parameters = {
    'tile shapes': [[16, 16], [16, 16], [16, 16], [8, 8]],
    'search regions': [[-1, 1], [-4, 4], [-4, 4], [-4, 4]]}

frames[1] = Image(frames[0].rgb_array)

# %% calculation
burst = ImageBurst(reference_image=frames[0],
                   images=frames[1:15],
                   downscale_factors=downscale_factors,
                   parameters=parameters)


@timer
def execu():
    burst.build_pyramids()
    burst.estimate_images_alignments()


execu()
# %% plot

compared_image = burst.images[1]
ref_image = burst.reference_image
plot_alignment(compared_image, ref_image)
