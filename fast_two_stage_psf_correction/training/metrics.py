from skimage import metrics
import numpy as np
from scipy import interpolate

import torch
import torch.nn.functional as F
import torch.nn as nn

from fast_optics_correction import raw2rgb
from fast_optics_correction import filters


def psnr(img1, img2, crop=0):
    if crop > 0:
        img1 = np.array(img1)[crop:-crop, crop:-crop]
        img2 = np.array(img2)[crop:-crop, crop:-crop]
    return metrics.peak_signal_noise_ratio(img1, img2)


def psnr_with_shifts(input, target, max_shift=2, interval_shift=0.5, crop=0):
    shift_x = np.arange(-max_shift, max_shift + interval_shift, interval_shift)
    shift_y = np.arange(-max_shift, max_shift + interval_shift, interval_shift)
    h, w = input.shape[0:2]
    X = np.arange(h)
    Y = np.arange(w)
    images = []
    if input.ndim == 3:
        multichannel = True
        f_red = interpolate.interp2d(X, Y, input[..., 0], kind='linear')
        f_green = interpolate.interp2d(X, Y, input[..., 1], kind='linear')
        f_blue = interpolate.interp2d(X, Y, input[..., 2], kind='linear')
        ssims = np.zeros(len(shift_x) * len(shift_y))
        i = 0
        crop = crop + max_shift
        for x in shift_x:
            for y in shift_y:
                X_shifted = X + x
                Y_shifted = Y + y
                input_shifted = np.stack([f_red(X_shifted, Y_shifted),
                                           f_green(X_shifted, Y_shifted),
                                           f_blue(X_shifted, Y_shifted)], axis=-1)
                input_shifted = input_shifted.astype(np.float32)
                images.append(input_shifted)
                ssims[i] = ssim(input_shifted, target, crop, multichannel=multichannel)
                i += 1
    else:
        multichannel = False
        f = interpolate.interp2d(X, Y, input, kind='linear')
        psnrs = np.zeros(len(shift_x) * len(shift_y))
        i = 0
        crop = crop + max_shift
        for x in shift_x:
            for y in shift_y:
                X_shifted = X + x
                Y_shifted = Y + y
                input_shifted = f(X_shifted, Y_shifted)
                input_shifted = input_shifted.astype(np.float32)
                images.append(input_shifted)
                psnrs[i] = psnr(input_shifted, target, crop)
                i += 1
    # print(ssims)
    idx_max = np.argmax(psnrs)
    psnr_max = np.max(psnrs)
    image_max = images[idx_max]
    # print(ssim_max)
    return psnr_max


def ssim(img1, img2, crop=0, multichannel=True):
    if crop > 0:
        img1 = np.array(img1)[crop:-crop, crop:-crop]
        img2 = np.array(img2)[crop:-crop, crop:-crop]
    return metrics.structural_similarity(img1, img2, multichannel=multichannel)


def ssim_with_shifts(input, target, max_shift=2, interval_shift=0.5, crop=0):
    shift_x = np.arange(-max_shift, max_shift + interval_shift, interval_shift)
    shift_y = np.arange(-max_shift, max_shift + interval_shift, interval_shift)
    h, w = input.shape[0:2]
    X = np.arange(h)
    Y = np.arange(w)
    images = []
    if input.ndim == 3:
        multichannel = True
        f_red = interpolate.interp2d(X, Y, input[..., 0], kind='linear')
        f_green = interpolate.interp2d(X, Y, input[..., 1], kind='linear')
        f_blue = interpolate.interp2d(X, Y, input[..., 2], kind='linear')
        ssims = np.zeros(len(shift_x) * len(shift_y))
        i = 0
        crop = crop + max_shift
        for x in shift_x:
            for y in shift_y:
                X_shifted = X + x
                Y_shifted = Y + y
                input_shifted = np.stack([f_red(X_shifted, Y_shifted),
                                           f_green(X_shifted, Y_shifted),
                                           f_blue(X_shifted, Y_shifted)], axis=-1)
                input_shifted = input_shifted.astype(np.float32)
                images.append(input_shifted)
                ssims[i] = ssim(input_shifted, target, crop, multichannel=multichannel)
                i += 1
    else:
        multichannel = False
        f = interpolate.interp2d(X, Y, input, kind='linear')
        ssims = np.zeros(len(shift_x) * len(shift_y))
        i = 0
        crop = crop + max_shift
        for x in shift_x:
            for y in shift_y:
                X_shifted = X + x
                Y_shifted = Y + y
                input_shifted = f(X_shifted, Y_shifted)
                input_shifted = input_shifted.astype(np.float32)
                images.append(input_shifted)
                ssims[i] = ssim(input_shifted, target, crop, multichannel=multichannel)
                i += 1
    # print(ssims)
    idx_max = np.argmax(ssims)
    ssim_max = np.max(ssims)
    image_max = images[idx_max]
    # print(ssim_max)
    return ssim_max, image_max


def error_ratio(input_non, input_gau, target, crop=25, with_shifts=True, interval_shift=1, max_shift=2):
    if with_shifts:
        ssim_non, input_non_shifted = ssim_with_shifts(input_non, target, crop=crop,
                                                       interval_shift=interval_shift, max_shift=max_shift)
        ssim_gau, input_gau_shifted = ssim_with_shifts(input_gau, target, crop=crop,
                                                       interval_shift=interval_shift, max_shift=max_shift)
        return (ssim_non + 2.0) / (ssim_gau + 2.0), input_non_shifted, input_gau_shifted
    else:
        ssim_non = ssim(input_non, target, crop=0)
        ssim_gau = ssim(input_gau, target, crop=0)
        return (ssim_non + 2.0) / (ssim_gau + 2.0)


def l1_with_isp(input, target, ccm):
    # process input
    input = raw2rgb.apply_ccm(input, ccm)
    input = raw2rgb.gamma_compression(input)
    # process target
    target = raw2rgb.apply_ccm(target, ccm)
    target = raw2rgb.gamma_compression(target)
    # evaluation
    return F.l1_loss(input, target)


class L1Grad(nn.Module):
    def __init__(self):
        super(L1Grad, self).__init__()

    def fast_grad(self, image):
        image_h = torch.roll(image, shifts=1, dims=-2)
        grad_h = image - image_h
        grad_h = grad_h[..., 1:, :]
        image_v = torch.roll(image, shifts=1, dims=-1)
        grad_v = image - image_v
        grad_v = grad_v[..., :, 1:].permute(0, 1, 3, 2)
        return torch.cat([grad_h, grad_v], dim=1)

    def forward(self, prediction, target):
        return F.l1_loss(self.fast_grad(prediction),  self.fast_grad(target))


class L1WithGamma(nn.Module):
    def __init__(self, gamma=2.2):
        super(L1WithGamma, self).__init__()
        self.gamma = gamma

    def forward(self, prediction, target):
        prediction = raw2rgb.gamma_compression(prediction.clamp(0, 1), gamma=self.gamma)
        target = raw2rgb.gamma_compression(target.clamp(0, 1), gamma=self.gamma)
        return F.l1_loss(prediction, target)
