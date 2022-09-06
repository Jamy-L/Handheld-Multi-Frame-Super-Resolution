import os
import rawpy
import matplotlib.pyplot as plt
import time
from skimage.io import imsave

import torch
import numpy as np

from fast_optics_correction import OpticsCorrection
from fast_optics_correction import utils


def read_image(impath):
    # make the difference between reading raw (DNG format only) or SRGB image
    if impath.split('.')[-1] == 'dng':
        raw = rawpy.imread(impath)
        raw_img = raw.postprocess(gamma=(1.0, 1.0), output_bps=16, use_camera_wb=True)
        img = raw_img.astype(np.float32) / (2 ** 16 - 1)
    else:
        img = plt.imread(impath)
        img = img.astype(np.float32) / 255
        img = img ** 2.2   # Linearize the srgb image
    return img


## Parameters
c = 0.416
sigma_b = 0.358
patch_size = 400
overlap_percentage = 0.25
ker_size = 31
polyblur_iteration = 1  # can be also 2 or 3
alpha = 2; b = 4
# alpha = 1; b = 6
# alpha = 3; b = 6
batch_size = 30
do_decomposition = False  # should we do a base/detail decomp. for not enhancing noise and artifacts?

device = torch.device('cuda:0')
print('Will run on', device)

## Read the image
name = 'facade'; impath = './pictures/facade.jpg'
# name = 'bridge'; impath = './pictures/bridge.jpg'
img = read_image(impath)

## Load the model
model = OpticsCorrection(patch_size=patch_size, overlap_percentage=overlap_percentage,
                         ker_size=ker_size)
model = model.to(device)


## Inference
img = utils.to_tensor(img).unsqueeze(0).to(device)
tic = time.time()
img_corrected = model(img, batch_size=batch_size, sigma_b=sigma_b, c=c, 
                      polyblur_iteration=polyblur_iteration, alpha=alpha, b=b,
                      do_decomposition=do_decomposition)
tac = time.time()
img_corrected = utils.to_array(img_corrected.cpu())
img = utils.to_array(img.cpu())
print('Restoration took %2.2f seconds.' % (tac - tic))


## Gamma curve as simple ISP
img = img ** (1./2.2)
img_corrected = img_corrected ** (1./2.2)


## Saving the images
savefolder = './results/'
os.makedirs(savefolder, exist_ok=True)

imsave(os.path.join(savefolder, '%s_original.png' % name), utils.to_uint(img))
imsave(os.path.join(savefolder, '%s_corrected.png' % name), utils.to_uint(img_corrected))
