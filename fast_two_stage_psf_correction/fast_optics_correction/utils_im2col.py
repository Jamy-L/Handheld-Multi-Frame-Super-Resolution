import numpy as np
import torch
import torch.nn.functional as F

from .utils import to_tensor, to_array


""" Im2col routines with simpler input arguments. """


#######################################
######## high-level routines ##########
#######################################


def im2col(input, patch_size=None, n_blocks=None, overlap_percentage=0.0, pad_border=True, interpolation_mode='constant'):
    """Wrapper to im2col that takes percentage of overlap as input instead of stride.
    images: *xCxHxW or HxWxC
    patches: *xLxCxpHxpW or LxpHxpWxC
    """
    if type(input) == np.ndarray:
        images = to_tensor(input).unsqueeze(0)
    else:
        images = input
    B = np.prod(images.shape[:-3])
    C, H, W = images.shape[-3:]
    ## Compute the stride and padding
    if patch_size is None:
        assert(n_blocks is not None)
        ## Slack dimension to make sure n_blocks devides H, W
        diff_H = np.ceil(H / n_blocks[0]) * n_blocks[0] - H
        diff_W = np.ceil(W / n_blocks[1]) * n_blocks[1] - W
        ## Ensure that dimension of image is coherent with blocks wanted
        pH = (H + diff_H) / n_blocks[0]  # first make a first estimate of pH
        pW = (W + diff_W) / n_blocks[1]
        ##
        pH = int(pH + 2*overlap_percentage * pH)  # take into account the overlap to make the patch bigger
        pW = int(pW + 2*overlap_percentage * pW)
        stride_H = pH * (1-overlap_percentage)  # compute the stride
        stride_W = pW * (1-overlap_percentage)
        if pad_border:
            pH = int(pH + 2 * pH * (overlap_percentage))
            pW = int(pW + 2 * pW * (overlap_percentage))
        new_H = int((n_blocks[0]-1)*stride_H + pH)
        new_W = int((n_blocks[1]-1)*stride_W + pW)
        patch_size = (pH, pW)
        stride_H = int(stride_H)
        stride_W = int(stride_W)
        stride = (stride_H, stride_W)
    else:
        assert(patch_size is not None)
        ## compute the stride
        pH, pW = patch_size
        stride_H = int(pH * (1 - overlap_percentage))
        stride_W = int(pW * (1 - overlap_percentage))
        stride = (stride_H, stride_W)
        ## compute first how many times moving the patch window to cover a lower bound of H and W
        n_patch_H = np.ceil(np.ceil((H-pH) / stride_H))
        n_patch_W = np.ceil(np.ceil((W-pW) / stride_W))
        ## now find the new values of H and W
        new_H = pH + n_patch_H * stride_H
        new_W = pW + n_patch_W * stride_W
        if pad_border:
            new_H += 2 * pH * overlap_percentage
            new_W += 2 * pW * overlap_percentage
            ## We have to recompute the number of strides
            n_patch_H = np.ceil(np.ceil((new_H-pH) / stride_H))
            n_patch_W = np.ceil(np.ceil((new_W-pW) / stride_W))
            ## Update accordingly the new image dimensions
            new_H = pH + n_patch_H * stride_H
            new_W = pW + n_patch_W * stride_W
        new_H = int(new_H)
        new_W = int(new_W)
    ## Compute then the padding for each border
    padding = get_padding_from_old_and_new_sizes((H, W), (new_H, new_W))
    ## Pad the images
    images = F.pad(images.reshape(B,C,H,W), padding, mode=interpolation_mode)
    padded_image_size = images.shape[-2:]
    ## Image to patch
    patches = im2ten(images, patch_size, stride)
    ## Reshape everything
    pH, pW = patch_size
    patches = patches.reshape(*images.shape[:-3], -1, C, pH, pW)
    if type(input) == np.ndarray:
        patches = patches.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    return patches, padded_image_size, patch_size, padding, stride


def col2im(input, padded_image_size, patch_size, padding, stride, mode='mean', window_type=None):
    """Wrapper to col2im that takes percentage of overlap as input instead of stride.
    patches: *xLxCxpHxpW
    images: *xCxHxW
    """
    assert(mode in ['sum', 'mean'])
    assert(window_type in [None, 'kaiser', 'hamming', 'bartlett', 'hann'])
    if type(input) == np.ndarray:
        patches = torch.from_numpy(input).permute(0, 3, 1, 2).unsqueeze(0)
    else:
        patches = input
    ## Get shapes
    # H, W = padded_image_size
    C = patches.shape[-3]
    B = np.prod(patches.shape[:-4])
    ## Add window to remove fusion artifact (optional)
    if window_type is not None:
        window = build_window(patch_size, window_type).to(patches.device)
        patches = patches * window
    else:
        window = torch.ones(patch_size).to(patches.device)
    ## Patches to image
    images = ten2im(patches.reshape(B, -1, C, *patch_size), padded_image_size, patch_size, stride, mode, window)
    ## Remove padding
    images = crop(images, padding)
    H, W = images.shape[-2:]
    ## Reshape everything
    images = images.reshape(*patches.shape[:-4], C, H, W)
    if type(input) == np.ndarray:
        images = to_array(images.cpu())
    return images


#######################################
######## low-level routines ###########
#######################################


def im2ten(image, patch_size, stride=(1,1)):
    """Converts an image into patches.
    image: NxCxHxW
    patches: NxLxCxpHxpW
    """
    b, c, h, w = image.shape
    hs, ws = patch_size
    patches = F.unfold(image, kernel_size=patch_size, stride=stride)
    patches = patches.permute(0,2,1).reshape(b,-1,c,hs,ws)
    return patches


def ten2im(patches, image_size, patch_size, stride=(1,1), mode='sum', window=None):
    """Converts a set of patches into an image.
    patches: NxLxCxpHxpW
    image: NxCxHxW
    """
    assert(len(window.shape) == 2 or window is None)
    b, n, c, hs, ws = patches.shape
    if mode == 'mean':
        if window is None:
            window = torch.ones(c, patch_size[0], patch_size[1], device=patches.device)  # c x pH x pW
        else:
            window = torch.stack([window for _ in range(c)], dim=0)  # c x pH x pW
        window = torch.stack([window for _ in range(n)], dim=0)  # n x c x pH x pW
        window = window.reshape(1, n, -1).permute(0, 2, 1)
        window = F.fold(window, output_size=image_size, kernel_size=patch_size, stride=stride)
    patches = patches.reshape(b, n, -1).permute(0, 2, 1)
    image = F.fold(patches, output_size=image_size, kernel_size=patch_size, stride=stride)
    if mode == 'mean':
        image = image / window
    return image


def im2arr(image, patch_size, stride=(1,1)):
    """Numpy wrapper for im2ten.
    images: HxWxC
    patches: LxpHxpWxC
    """
    image = to_tensor(image).unsqueeze(0)
    patches = im2ten(image, patch_size, stride)
    patches = patches.squeeze(0).permute(0,2,3,1).numpy()
    return patches


def arr2im(patches, image_size, patch_size, stride=(1,1), mode='sum'):
    """Numpy wrapper for ten2im.
    patches: LxpHxpWxC
    images: HxWxC
    """
    # to tensor
    patches = torch.from_numpy(patches).permute(0,3,1,2).unsqueeze(0).contiguous()
    image = ten2im(patches, image_size, patch_size, stride, mode)
    image = to_array(image)
    # image is HxWxC np.ndarray
    return image


def get_padding_from_old_and_new_sizes(old_size, new_size):
    H, W = old_size
    new_H, new_W = new_size
    diff_H = new_H - H
    diff_W = new_W - W
    padding_left = int(np.floor(diff_W/2))
    padding_right = int(np.ceil(diff_W/2))
    padding_top = int(np.floor(diff_H/2))
    padding_bottom = int(np.ceil(diff_H/2))
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    return padding


def crop(images, padding):
    padding_left, padding_right, padding_top, padding_bottom = padding
    if padding_left > 0:
        images = images[..., padding_left:]
    if padding_right > 0:
        images = images[..., :-padding_right]
    if padding_top > 0:
        images = images[..., padding_top:, :]
    if padding_bottom > 0:
        images = images[..., :-padding_bottom, :]
    return images


def build_window(image_size, window_type):
    H, W = image_size
    if window_type == 'kaiser':
        window_i = torch.kaiser_window(H, beta=5, periodic=False)
        window_j = torch.kaiser_window(W, beta=5, periodic=False)
    elif window_type == 'hann':
        window_i = torch.hann_window(H, periodic=False)
        window_j = torch.hann_window(W, periodic=False)
    elif window_type == 'hamming':
        window_i = torch.hamming_window(H, periodic=False)
        window_j = torch.hamming_window(W, periodic=False)
    elif window_type == 'bartlett':
        window_i = torch.bartlett_window(H, periodic=False)
        window_j = torch.bartlett_window(W, periodic=False)
    else:
        Exception('Window not implemented')

    return window_i.unsqueeze(-1) * window_j.unsqueeze(0)
