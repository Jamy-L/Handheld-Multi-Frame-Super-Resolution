import numpy as np
import torch
import torch.nn.functional as F
import torch.fft
from skimage import img_as_float32


def to_tensor(x, to_type=torch.float):
    """Converts an nd.array into a torch.tensor of size (C,H,W)."""
    ndim = len(x.shape)
    if ndim == 2:
        x = torch.from_numpy(x.copy()).unsqueeze(0)
    else:
        x = torch.from_numpy(x.copy()).permute(2, 0, 1)
    if to_type == torch.float:
        x = x.float()
    elif to_type == torch.double:
        x = x.double()
    else:
        x = x.long()
    return x


def to_array(x):
    """Converts a torch.tensor into an nd.array of size (H,W,C)."""
    x = x.squeeze().detach().cpu()
    if len(x.shape) == 2:
        return x.numpy()
    else:
        x = x.permute(1, 2, 0)
        return x.numpy()
    
    
def to_float(img):
    """Converts an ndarray to np.float32 type."""
    img = img_as_float32(img)
    img = img.astype(np.float32)
    return img


def to_uint(img):
    """Converts an ndarray to np.uint8 type."""
    img = img_as_float32(img)
    img = (255*img).astype(np.uint8)
    return img


def rgb_to_raw(rgb):
    """Simple (H,W,3) RGB image conversion to (H/2,W/2,4) raw format with RGGB pattern."""
    flag_array = False
    if type(rgb) == np.ndarray:
        flag_array = True
        rgb = to_tensor(rgb).unsqueeze(0)
    b, _, h, w = rgb.shape
    raw = torch.zeros(b, 4, h//2, w//2, device=rgb.device)
    raw[:, 0] = rgb[:, 0, 0::2, 0::2]
    raw[:, 1] = rgb[:, 1, 0::2, 1::2]
    raw[:, 2] = rgb[:, 1, 1::2, 0::2]
    raw[:, 3] = rgb[:, 2, 1::2, 1::2]
    if flag_array:
        raw = to_array(raw)
    return raw


mosaic = rgb_to_raw


def mosaicked_to_raw(mosaicked):
    """Transforms the (H,W) image to (H/2,W/2,4) image in RGGB pattern."""
    flag_array = False
    if type(mosaicked) == np.ndarray:
        flag_array = True
        mosaicked = to_tensor(mosaicked).unsqueeze(0)
    b, _, h, w = mosaicked.shape
    raw = torch.zeros(b, 4, h//2, w//2, device=mosaicked.device)
    raw[:, 0] = mosaicked[..., 0::2, 0::2]
    raw[:, 1] = mosaicked[..., 0::2, 1::2]
    raw[:, 2] = mosaicked[..., 1::2, 0::2]
    raw[:, 3] = mosaicked[..., 1::2, 1::2]
    if flag_array:
        raw = to_array(raw)
    return raw


def raw_to_mosaicked(raw):
    """Transforms the (H/2,W/2,4) image to (H,W) image in RGGB pattern."""
    flag_array = False
    if type(raw) == np.ndarray:
        flag_array = True
        raw = to_tensor(raw).unsqueeze(0)
    b, _, h, w = raw.shape
    mosaicked = torch.zeros(b, 1, 2*h, 2*w).type_as(raw)
    mosaicked[..., 0::2, 0::2] = raw[:, 0:1]
    mosaicked[..., 0::2, 1::2] = raw[:, 1:2]
    mosaicked[..., 1::2, 0::2] = raw[:, 2:3]
    mosaicked[..., 1::2, 1::2] = raw[:, 3:4]
    if flag_array:
        mosaicked = to_array(mosaicked)
    return mosaicked


def raw_to_atrous(raw):
    """Transforms the (H/2,W/2,4) image to (H,W,3) image in RGGB pattern."""
    flag_array = False
    if type(raw) == np.ndarray:
        flag_array = True
        raw = to_tensor(raw).unsqueeze(0)
    b, _, h, w = raw.shape
    atrous = torch.zeros(b, 3, 2*h, 2*w, device=raw.device)
    atrous[:, 0, 0::2, 0::2] = raw[:, 0]
    atrous[:, 1, 0::2, 1::2] = raw[:, 1]
    atrous[:, 1, 1::2, 0::2] = raw[:, 2]
    atrous[:, 2, 1::2, 1::2] = raw[:, 3]
    if flag_array:
        atrous = to_array(atrous)
    return atrous


def atrous_to_raw(atrous):
    """Transforms the (H,W,3) image to (H/2,W/2,4) image in RGGB pattern."""
    flag_array = False
    if type(atrous) == np.ndarray:
        flag_array = True
        atrous = to_tensor(atrous).unsqueeze(0)
    b, _, h, w = atrous.shape
    raw = torch.zeros(b, 4, h//2, w//2, device=atrous.device)
    raw[:, 0] = atrous[:, 0, 0::2, 0::2]
    raw[:, 1] = atrous[:, 1, 0::2, 1::2]
    raw[:, 2] = atrous[:, 1, 1::2, 0::2]
    raw[:, 3] = atrous[:, 2, 1::2, 1::2]
    if flag_array:
        raw = to_array(raw)
    return raw


def mosaicked_to_atrous(mosaicked):
    """Transorms the (H,W) image to (H/2,W/2,3) image in RGGB pattern."""
    raw = mosaicked_to_raw(mosaicked)
    atrous = raw_to_atrous(raw)
    return atrous


def atrous_to_mosaicked(atrous):
    """Transforms the (H,W,3) image to (H,W) image in RGGB pattern."""
    raw = atrous_to_raw(atrous)
    mosaicked = raw_to_mosaicked(raw)
    return mosaicked


def depth_to_space(x, r):
    """Pytorch version of tf.image.depth_to_space."""
    return F.pixel_shuffle(x, r)


def space_to_depth(x, r):
    """Pytorch version of tf.image.space_to_depth. Inverse routine to F.pixel_shuffle."""
    return F.pixel_unshuffle(x, r)


def pad_with_new_size(img, new_size, mode='constant'):
    h, w = img.shape[-2:]
    new_h, new_w = new_size
    pad_left = int(np.floor((new_w - w)/2))
    pad_right = int(np.ceil((new_w - w) / 2))
    pad_top = int(np.floor((new_h - h)/2))
    pad_bottom = int(np.ceil((new_h - h) / 2))
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    img = F.pad(img, padding, mode=mode)
    return img


def crop_with_old_size(img, old_size):
    h, w, = img.shape[-2:]
    old_h, old_w = old_size
    crop_left = int(np.floor((w - old_w)/2))
    if crop_left > 0:
        img = img[..., :, crop_left:]
    crop_right = int(np.ceil((w - old_w) / 2))
    if crop_right > 0:
        img = img[..., :, :-crop_right]
    crop_top = int(np.floor((h - old_h) / 2))
    if crop_top > 0:
        img = img[..., crop_top:, :]
    crop_bottom = int(np.ceil((h - old_h) / 2))
    if crop_bottom > 0:
        img = img[..., :-crop_bottom, :]
    return img
