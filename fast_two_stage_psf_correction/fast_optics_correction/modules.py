import torch
import torch.nn as nn
import numpy as np
import os

from . import basicblock as B
from .polyblur_functions import mild_inverse_rank3, blur_estimation, recursive_filter
from .utils import pad_with_new_size, crop_with_old_size



class OpticsCorrection(nn.Module):
    def __init__(self, load_weights=True, model_type='tiny', patch_size=400, overlap_percentage=0.25, ker_size=31):
        super(OpticsCorrection, self).__init__()
        ## Sharpening attributes
        self.patch_size = patch_size
        self.overlap_percentage = overlap_percentage
        self.ker_size = ker_size

        ## Defringing attributes
        if model_type == 'tiny':
            self.defringer = ResUNet(nc=[16, 32, 64, 64], in_nc=2, out_nc=1)
        elif model_type == 'super_tiny':
            self.defringer = ResUNet(nc=[16, 16, 32, 32], in_nc=2, out_nc=1)
        else:
            self.defringer = ResUNet(nc=[64, 128, 256, 512], in_nc=2, out_nc=1)
        if load_weights:
            state_dict_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../checkpoints/' + model_type + '_epoch_1000.pt')
            self.defringer.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
            self.defringer.eval()


    def forward(self, image, batch_size=20, c=0.358, sigma_b=0.451, polyblur_iteration=1, alpha=2, b=3, 
                do_decomposition=False, do_halo_removal=False):
        assert(image.shape[0] == 1)  # One image at the time

        ## Make sure dimensions are even
        image, h, w = self.make_even_dimensions(image)
 
        ## Pad the image if needed
        img_padded, new_h, new_w = self.pad_image(image)
 
        ## Get indices of the top-left corners of the patches
        n_blocks, IJ_coords = self.get_patch_indices(new_h, new_w)

        ## Create the arrays for outputing results
        ps = self.patch_size
        restored_image = torch.zeros_like(img_padded)  # (1,3,H,W)
        window = self.build_window(ps, window_type='kaiser').unsqueeze(0).unsqueeze(0).to(image.device)  # (1,1,pH,pW)
        window_sum = torch.zeros((1, 1, new_h, new_w)).to(image.device)  # (1,1,H,W)

        ## Main loop on patches
        n_chuncks = int(np.ceil(n_blocks / batch_size))
        for n in range(n_chuncks):
            ## Extract the patch
            IJ_coords_batch = IJ_coords[n*batch_size:(n+1)*batch_size]  # (b,2)
            patch = [img_padded[..., i:i + ps, j:j + ps] for (i, j) in IJ_coords_batch]
            patch = torch.cat(patch, dim=0)  # (b,3,pH,pW)

            ##### Blind deblurring module (Polyblur)
            for _ in range(polyblur_iteration):
                if do_decomposition:
                    patch_base = recursive_filter(patch, sigma_s=2, sigma_r=0.1)
                    patch_detail = patch - patch_base
                    kernel = blur_estimation(patch_base, c=c, sigma_b=sigma_b, ker_size=self.ker_size)
                    patch_base = mild_inverse_rank3(patch_base, kernel, correlate=True, halo_removal=False, alpha=alpha, b=b)  # (b,3,pH,pW)
                    patch = patch_detail + patch_base
                else:
                    kernel = blur_estimation(patch, c=c, sigma_b=sigma_b, ker_size=self.ker_size)
                    patch = mild_inverse_rank3(patch, kernel, correlate=True, halo_removal=False, alpha=alpha, b=b)  # (b,3,pH,pW)
                patch = patch.clamp(0, 1)

            ##### Defringing module
            ## Extract indiviual color channels
            patch_red = patch[:, 0:1]
            patch_green = patch[:, 1:2]
            patch_blue = patch[:, 2:3]

            ## Inference
            with torch.no_grad():
                patch_red_restored = patch_red - self.defringer(torch.cat([patch_red, patch_green], dim=1))
                patch_blue_restored = patch_blue - self.defringer(torch.cat([patch_blue, patch_green], dim=1))

            ## Replace the patches and weights
            for m in range(IJ_coords_batch.shape[0]):
                i, j = IJ_coords_batch[m]
                restored_image[:, 0:1, i:i + ps, j:j + ps] += window * patch_red_restored[m]
                restored_image[:, 1:2, i:i + ps, j:j + ps] += window * patch_green[m]
                restored_image[:, 2:3, i:i + ps, j:j + ps] += window * patch_blue_restored[m]
                window_sum[..., i:i + ps, j:j + ps] += window

        ## Normalize and crop the final image
        restored_image = self.postprocess_result(restored_image, window_sum, (h, w))

        return restored_image


    def make_even_dimensions(self, image):
        h, w = image.shape[-2:]
        if h % 2 == 1:
            image = image[..., :-1, :]
            h -= 1
        if w % 2 == 1:
            image = image[..., :, :-1]
            w -= 1
        return image, h, w


    def pad_image(self, image):
        h, w = image.shape[-2:]
        new_h = int(np.ceil(h / self.patch_size) * self.patch_size)
        new_w = int(np.ceil(w / self.patch_size) * self.patch_size)
        img_padded = pad_with_new_size(image, (new_h, new_w), mode='replicate')
        return img_padded, new_h, new_w


    def postprocess_result(self, restored_image, window_sum, old_size):
        restored_image = restored_image / (window_sum + 1e-8)
        restored_image = crop_with_old_size(restored_image, old_size)
        restored_image = torch.clamp(restored_image, 0.0, 1.0)
        return restored_image


    def get_patch_indices(self, new_h, new_w):
        I_coords = np.arange(0, new_h - self.patch_size + 1, int(self.patch_size * (1 - self.overlap_percentage)))
        J_coords = np.arange(0, new_w - self.patch_size + 1, int(self.patch_size * (1 - self.overlap_percentage)))
        IJ_coords = np.meshgrid(I_coords, J_coords, indexing='ij')
        IJ_coords = np.stack(IJ_coords).reshape(2, -1).T
        n_blocks = len(I_coords) * len(J_coords)
        return n_blocks, IJ_coords


    def build_window(self, image_size, window_type):
        H = W = image_size
        if window_type == 'kaiser':
            window_i = torch.kaiser_window(H, beta=5, periodic=True)
            window_j = torch.kaiser_window(W, beta=5, periodic=True)
        elif window_type == 'hann':
            window_i = torch.hann_window(H, periodic=True)
            window_j = torch.hann_window(W, periodic=True)
        elif window_type == 'hamming':
            window_i = torch.hamming_window(H, periodic=True)
            window_j = torch.hamming_window(W, periodic=True)
        elif window_type == 'bartlett':
            window_i = torch.bartlett_window(H, periodic=True)
            window_j = torch.bartlett_window(W, periodic=True)
        else:
            Exception('Window not implemented')

        return window_i.unsqueeze(-1) * window_j.unsqueeze(0)



"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


