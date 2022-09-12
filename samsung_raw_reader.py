# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import zipfile
import shutil


def load_txt(path):
    with open(path, 'r') as fh:
        out = [d.rstrip() for d in fh.readlines()]

    return out


class SamsungRAWImage:
    """ Custom class for RAW images captured from Samsung Galaxy S8 """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return SamsungRAWImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
                               meta_data['daylight_wb'], meta_data['color_matrix'], meta_data['exif_data'],
                               meta_data.get('im_preview', None))

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, color_matrix, exif_data, im_preview=None):
        self.im_raw = im_raw
        self.black_level = black_level
        self.cam_wb = cam_wb
        self.daylight_wb = daylight_wb
        self.color_matrix = color_matrix
        self.exif_data = exif_data
        self.im_preview = im_preview

        self.norm_factor = 1023.0

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'color_matrix': self.color_matrix.tolist()}

    def get_exposure_time(self):
        return self.exif_data['Image ExposureTime'].values[0].decimal()

    def get_noise_profile(self):
        noise = self.exif_data['Image Tag 0xC761'].values
        noise = [n[0] for n in noise]
        noise = np.array(noise).reshape(3, 2)
        return noise

    def get_f_number(self):
        return self.exif_data['Image FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['Image ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        if self.im_preview is not None:
            im_preview = self.im_preview[2*r1:2*r2, 2*c1:2*c2]
        else:
            im_preview = None

        return SamsungRAWImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.color_matrix,
                               self.exif_data, im_preview=im_preview)

    def postprocess(self, return_np=True, norm_factor=None):
        raise NotImplementedError


class CanonImage:
    """ Custom class for RAW images captured from Canon DSLR """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return CanonImage(im_raw.float(), meta_data['black_level'], meta_data['cam_wb'],
                          meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'])

    @staticmethod
    def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                                 no_white_balance=False):
        im = im * meta_data.get('norm_factor', 1.0)

        if not meta_data.get('black_level_subtracted', False):
            im = (im - torch.tensor(meta_data['black_level'])[[0, 1, -1]].view(3, 1, 1))

        if not meta_data.get('while_balance_applied', False) and not no_white_balance:
            im = im * torch.tensor(meta_data['cam_wb'])[[0, 1, -1]].view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

        im_out = im

        if external_norm_factor is None:
            im_out = im_out / (im_out.mean() * 5.0)
        else:
            im_out = im_out / external_norm_factor

        im_out = im_out.clamp(0.0, 1.0)

        if gamma:
            im_out = im_out ** (1.0 / 2.2)

        if smoothstep:
            # Smooth curve
            im_out = 3 * im_out ** 2 - 2 * im_out ** 3

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, rgb_xyz_matrix, exif_data):
        super(CanonImage, self).__init__()
        self.im_raw = im_raw

        if len(black_level) == 4:
            black_level = [black_level[0], black_level[1], black_level[3]]
        self.black_level = black_level

        if len(cam_wb) == 4:
            cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
        self.cam_wb = cam_wb

        if len(daylight_wb) == 4:
            daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
        self.daylight_wb = daylight_wb

        self.rgb_xyz_matrix = rgb_xyz_matrix

        self.exif_data = exif_data

        self.norm_factor = 16383

    def shape(self):
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'rgb_xyz_matrix': self.rgb_xyz_matrix.tolist(), 'norm_factor': self.norm_factor}

    def get_exposure_time(self):
        return self.exif_data['EXIF ExposureTime'].values[0].decimal()

    def get_f_number(self):
        return self.exif_data['EXIF FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['EXIF ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def set_image_data(self, im_data):
        self.im_raw = im_data

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return CanonImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.rgb_xyz_matrix,
                          self.exif_data)

    def set_crop_info(self, crop_info):
        self.crop_info = crop_info

    def resize(self, size=None, scale_factor=None):
        self.im_raw = F.interpolate(self.im_raw.unsqueeze(0), size=size, scale_factor=scale_factor,
                                    mode='bilinear').squeeze(0)

    def postprocess(self, return_np=True):
        raise NotImplementedError



def raw_to_array(path):
    im_tensor = SamsungRAWImage.load(path).im_raw
    array = np.empty((im_tensor.shape[1]*2, im_tensor.shape[2]*2))
    # output format : B 00, G 01 10, R 11
    array[::2,::2] = im_tensor[3]
    array[::2,1::2] = im_tensor[1]
    array[1::2,::2] = im_tensor[2]
    array[1::2,1::2] = im_tensor[0]
    return array
    
        
        
