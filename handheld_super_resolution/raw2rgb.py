import torch
import numpy as np
import random
from skimage import img_as_float32, filters

import math


def get_random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
               [-0.5625, 1.6328, -0.0469],
               [-0.0703, 0.2188, 0.6406]],
              [[0.4913, -0.0541, -0.0202],
               [-0.613, 1.3513, 0.2906],
               [-0.1564, 0.2151, 0.7183]],
              [[0.838, -0.263, -0.0639],
               [-0.2887, 1.0725, 0.2496],
               [-0.0627, 0.1427, 0.5438]],
              [[0.6596, -0.2079, -0.0562],
               [-0.4782, 1.3016, 0.1933],
               [-0.097, 0.1581, 0.5181]]]

    num_ccms = len(xyz2cams)
    xyz2cams = torch.tensor(xyz2cams)

    weights = torch.FloatTensor(num_ccms, 1, 1).uniform_(0.0, 1.0).numpy()
    weights_sum = weights.sum()
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]]).float()
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdims=True)
    return rgb2cam


def get_random_noise_parameters(log_min_shot=0.0001, log_max_shot=0.012, sigma_read_noise=0.26):
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(log_min_shot)
    log_max_shot_noise = math.log(log_max_shot)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=sigma_read_noise)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise


def get_random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / random.gauss(mu=0.8, sigma=0.1)

    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] == 3

    gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains.view(-1, 1, 1)

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = image.mean(dim=0, keepdims=True)
    inflection = 0.9
    mask = ((gray - inflection).clamp(0.0) / (1.0 - inflection)) ** 2.0

    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains


def apply_gains(image, red_gain, blue_gain, rgb_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] in [3, 4]

    if image.shape[0] == 3:
        gains = torch.tensor([red_gain, 1.0, blue_gain]) * rgb_gain
    else:
        gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]) * rgb_gain
    gains = gains.view(-1, 1, 1)
    gains = gains.to(image.device).type_as(image)

    return (image * gains).clamp(0.0, 1.0)


def get_color_matrix(raw):
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
    xyz2cam = raw.rgb_xyz_matrix[:3]
    rgb2cam = xyz2cam @ rgb2xyz

    # Normalizes each row.
    rgb2cam = rgb2cam / rgb2cam.sum(axis=-1, keepdims=True)
    return rgb2cam.astype(np.float32)


def apply_ccm(image, ccm):
    if type(image) == np.ndarray:
        assert(image.ndim == 3 and image.shape[-1] == 3)
        image = np.transpose(image, (2, 0, 1))
        shape = image.shape
        image = image.reshape(3, -1)
        image = np.matmul(ccm, image)
        image = image.reshape(shape)
        return np.transpose(image, (1, 2, 0))
    else:
        assert image.dim() == 3 and image.shape[0] == 3
        shape = image.shape
        image = image.view(3, -1)
        ccm = ccm.to(image.device).type_as(image)
        image = torch.matmul(ccm, image)
        return image.view(shape)


def gamma_compression(img, gamma=2.2):
    if type(img) == np.ndarray:
        return img**(1./gamma)
    else:
        return img.clamp(1e-8) ** (1 / gamma)


def gamma_expansion(img, gamma=2.2):
    if type(img) == np.ndarray:
        return img ** gamma
    else:
        return img.clamp(1e-8) ** gamma


def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    image_out = 3 * image**2 - 2 * image**3
    return image_out


def invert_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    if type(np.ndarray):
        image = np.clip(image, 0.0, 1.0)
        return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)
    else:
        image = image.clamp(0.0, 1.0)
        return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def unprocess_isp(jpg, log_max_shot=0.012):
    """
    Convert a jpg image to raw image.
    """
    rgb2cam = get_random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = get_random_gains()
    lambda_read, lambda_shot = get_random_noise_parameters(log_max_shot=log_max_shot)
    metadata = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                'blue_gain': blue_gain, 'lambda_shot': lambda_shot, 'lambda_read': lambda_read}

    ## Inverse tone mapping
    jpg = utils.to_tensor(jpg)
    jpg = invert_smoothstep(jpg)

    ## Gamma expansion
    jpg = gamma_expansion(jpg)

    ## Inverse color matrix
    raw = apply_ccm(jpg, rgb2cam)

    ## Inverse gains
    raw = safe_invert_gains(raw, red_gain, blue_gain, rgb_gain)
    raw = utils.to_array(raw)

    return raw, metadata


def process_isp(raw, img=None, do_color_correction=True, do_tonemapping=True, do_gamma=True, do_sharpening=True):
    """
    Convert a raw image to jpg image.
    """
    if img is None:
        ## Rawpy processing - whole stack
        return img_as_float32(raw.postprocess(use_camera_wb=True))
    else:
        ## Color matrix
        if do_color_correction:
            # rgb2cam = get_color_matrix(raw)
            rgb2cam = raw.color_matrix[:3, :3]
            cam2rgb = np.linalg.inv(rgb2cam)
            img = apply_ccm(img, cam2rgb)
            img = np.clip(img, 0.0, 1.0)
        ## Gamma compression
        if do_gamma:
            img = gamma_compression(img)
        ## Sharpening
        if do_sharpening:
            img = filters.unsharp_mask(img)
        ## Tone mapping
        if do_tonemapping:
            img = apply_smoothstep(img)
        img = np.clip(img, 0.0, 1.0)

        return img
