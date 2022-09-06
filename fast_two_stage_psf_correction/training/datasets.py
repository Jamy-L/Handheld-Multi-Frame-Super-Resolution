import random

import numpy as np
import torch
import torch.utils.data as data
import os
import glob
import tqdm

from skimage import io, img_as_float32, restoration
from scipy import ndimage

from fast_optics_correction import raw2rgb
from fast_optics_correction import filters
from fast_optics_correction import utils
from fast_optics_correction import utils_psf


def read_images(image_folder, training, load_images=False):
    if training:
        image_paths = sorted(glob.glob(os.path.join(image_folder, 'DIV2K_train_HR/*.png')))
    else:
        image_paths = sorted(glob.glob(os.path.join(image_folder, 'DIV2K_valid_HR/*.png')))
    if load_images:
        images = [img_as_float32(io.imread(path)) for path in tqdm.tqdm(image_paths, desc='Read images')]
        return images
    else:
        return image_paths


def read_psfs(psf_folder, training, n_psfs=100):
    psf_paths = sorted(glob.glob(os.path.join(psf_folder, '*.mat')))
    if training:
        psf_paths = psf_paths[1:]
    else:
        psf_paths = psf_paths[:1]
    psfs = []
    for path in tqdm.tqdm(psf_paths[:n_psfs], desc='Read PSFs'):
        psf = utils_psf.PSF(path)
        N = len(psf)
        H, W = psf.get_sensor_size()
        for n in range(N):
            location, _, kernel = psf.get_psf_by_index(n)
            i, j = location
            i_norm = 2 * (i / H) - 1  # normalized coordinates
            j_norm = 2 * (j / W) - 1
            psfs.append((kernel, i_norm, j_norm))
    return psfs


def crop(img, patch_size, training):
    H, W, _ = img.shape
    if training:
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))
        patch = img[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
    else:
        h = (H - patch_size) // 2
        w = (W - patch_size) // 2
        patch = img[h:h + patch_size, w:w + patch_size]
    return patch


def augment_img(img, mode=0):
    '''
    Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def get_gaussian_filter(training):
    if training:
        theta = random.random() * 2 * np.pi
        ## Red
        sigma_theta = (4-0.2) * random.random() + 0.2
        rho = (1-0.15) * random.random() + 0.15
        sigma_perp = max(sigma_theta * rho, 0.2)
        sigma = (sigma_theta, sigma_perp)
        mu_x = 8 * random.random() - 4
        mu_y = 8 * random.random() - 4
        mu_theta = np.cos(theta) * mu_x - np.sin(theta) * mu_y
        mu_perp = np.sin(theta) * mu_x + np.sin(theta) * mu_y
        mu = np.array([mu_theta, mu_perp])
        kernel_red = filters.gaussian_filter(sigma, theta, mu, k_size=np.array([31, 31]))
        ## Blue
        sigma_theta = (4 - 0.2) * random.random() + 0.2
        rho = (1 - 0.15) * random.random() + 0.15
        sigma_perp = max(sigma_theta * rho, 0.2)
        sigma = (sigma_theta, sigma_perp)
        mu_x = 8 * random.random() - 4
        mu_y = 8 * random.random() - 4
        mu_theta = np.cos(theta) * mu_x - np.sin(theta) * mu_y
        mu_perp = np.sin(theta) * mu_x + np.sin(theta) * mu_y
        mu = np.array([mu_theta, mu_perp])
        kernel_blue = filters.gaussian_filter(sigma, theta, mu, k_size=np.array([31, 31]))
        ## Green
        sigma_theta = (4-0.2) * random.random() + 0.2
        rho = (1-0.15) * random.random() + 0.15
        sigma_perp = max(sigma_theta * rho, 0.2)
        sigma = (sigma_theta, sigma_perp)
        mu_theta = 0
        mu_perp = 0
        mu = np.array([mu_theta, mu_perp])
        kernel_green = filters.gaussian_filter(sigma, theta, mu, k_size=np.array([31, 31]))
    else:
        theta = 0
        sigma_red = (3.3, 1.3)
        mu_red = np.array([2.3, 1.7])
        sigma_green = (2.7, 1.0)
        mu_green = np.array([0.0, -0.0])
        sigma_blue = (2.1, 0.7)
        mu_blue = np.array([-2.0, -1.7])
        kernel_red = filters.gaussian_filter(sigma_red, theta, mu_red, k_size=np.array([31, 31]))
        kernel_green = filters.gaussian_filter(sigma_green, theta, mu_green, k_size=np.array([31, 31]))
        kernel_blue = filters.gaussian_filter(sigma_blue, theta, mu_blue, k_size=np.array([31, 31]))
    return np.stack([kernel_red, kernel_green, kernel_blue], axis=-1)


def add_blur_and_noise(x, k, correlate=False, training=False, simulate_saturation=False, add_noise=False):
    # unprocess image
    x, metadata = raw2rgb.unprocess_isp(x, log_max_shot=0.002)  # Relatively ok
    lambda_shot = metadata['lambda_shot']
    lambda_read = metadata['lambda_read']

    if simulate_saturation and training and random.random() < 0.2:
        x = (1 + random.random()) * x

    # blur the image
    if correlate:
        y = [ndimage.filters.correlate(x[..., c], k[..., c], mode='wrap') for c in range(3)]
    else:
        y = [ndimage.filters.convolve(x[..., c], k[..., c], mode='wrap') for c in range(3)]
    y = np.stack(y, axis=-1)

    # go to pytorch
    x = utils.to_tensor(x)
    y = utils.to_tensor(y)
    k = utils.to_tensor(k)
    lambda_shot = torch.tensor(lambda_shot).view(1, 1, 1)
    lambda_read = torch.tensor(lambda_read).view(1, 1, 1)
    cam2rgb = torch.from_numpy(metadata['cam2rgb'])

    # compute and add noise
    if add_noise and training:
        n = torch.sqrt(lambda_shot * y + lambda_read) * torch.randn_like(y)
        y = y + n

    # process y
    y = raw2rgb.apply_gains(y, metadata['red_gain'], metadata['blue_gain'], metadata['rgb_gain'])
    y = torch.clamp(y, 0.0, 1.0)
    y = raw2rgb.apply_ccm(y, cam2rgb)
    y = torch.clamp(y, 0.0, 1.0)

    # process x
    x = raw2rgb.apply_gains(x, metadata['red_gain'], metadata['blue_gain'], metadata['rgb_gain'])
    x = torch.clamp(x, 0.0, 1.0)
    x = raw2rgb.apply_ccm(x, cam2rgb)
    x = torch.clamp(x, 0.0, 1.0)

    return x, y, k, lambda_read, lambda_shot, cam2rgb


class DatasetGaussianBlur(data.Dataset):
    def __init__(self, opts, training=False, use_gaussian_filters=False, n_psfs=100):
        super(DatasetGaussianBlur, self).__init__()

        self.training = training
        self.patch_size = opts.patch_size
        self.simulate_saturation = opts.simulate_saturation
        self.images = read_images(opts.images_folder, training=training, load_images=opts.load_images)
        self.load_images = opts.load_images
        if use_gaussian_filters:
            self.kernels = None
        else:
            self.kernels = read_psfs(opts.psfs_folder, training=training, n_psfs=n_psfs)
        self.do_denoising = opts.do_denoising

    def __getitem__(self, index):
        ## Read image
        if self.load_images:
            sharp = self.images[index % len(self.images)]
        else:
            img_path = self.images[index % len(self.images)]
            sharp = img_as_float32(io.imread(img_path))
        sharp = crop(sharp, self.patch_size, self.training)

        ## Image augmentation
        if self.training:
            mode = random.randint(0, 7)
            sharp = augment_img(sharp, mode=mode)

        ## Read psf
        if self.kernels is None:
            kernel = get_gaussian_filter(self.training)
        else:
            if self.training:
                index_psf = random.randint(0, len(self.kernels) - 1)
            else:
                index_psf = index % len(self.kernels)
            # theta, sigma, mean, i_norm, j_norm = self.psfs[index_psf]
            kernel, _, _ = self.kernels[index_psf]

        ## Blur and noise the image
        outputs = add_blur_and_noise(sharp, kernel, correlate=True, training=self.training,
                                     simulate_saturation=self.simulate_saturation, add_noise=self.do_denoising)
        sharp = outputs[0]
        blurry = outputs[1]
        kernel = outputs[2]

        ## (Optional) Denoise the image
        if self.do_denoising:
            blurry = utils.rgb_to_raw(blurry.unsqueeze(0))  # 1x4xH/2xW/2
            blurry = utils.to_array(blurry)  # HxWx4
            blurry[..., 0] = restoration.denoise_bilateral(blurry[..., 0], win_size=7)
            blurry[..., 1] = restoration.denoise_bilateral(blurry[..., 1], win_size=7)
            blurry[..., 2] = restoration.denoise_bilateral(blurry[..., 2], win_size=7)
            blurry[..., 3] = restoration.denoise_bilateral(blurry[..., 3], win_size=7)
            blurry = utils.to_tensor(blurry)  # 4xH/2xW/2
        return sharp, blurry, kernel

    def __len__(self):
        return len(self.images)

