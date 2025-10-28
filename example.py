# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:10:18 2023

This is an example script showing how to call the pipeline and specify
the parameters.

Make sure that the example bursts have been downloaded from the release version!!


@author: jamyl
"""

import os
import matplotlib.pyplot as plt
from handheld_super_resolution import process
from skimage import img_as_ubyte
from omegaconf import OmegaConf

# Load the default configuration
default_conf = OmegaConf.load("configs/default.yaml")

# Set your config like that.
my_custom_conf = OmegaConf.create({
    "verbose": 2, 
    "scale": 2,
    "merging" : {'kernel': 'steerable'},
    'post processing' : {'enabled':True}
})
# Alterbnatively, put this custom conf in a yaml file and load it with:
# my_custom_conf = OmegaConf.load("path_to_your_custom_config.yaml")

# Merge user config over the default config
config = OmegaConf.merge(default_conf, my_custom_conf)

# calling the pipeline
burst_path = './test_burst/Samsung/'
output_img = process(burst_path, config)[0]

# saving the result
os.makedirs('./results', exist_ok=True)
plt.imsave('./results/output_img.png', img_as_ubyte(output_img))


# plotting the result
plt.figure("output")
plt.imshow(output_img, interpolation = 'none')
plt.xticks([])
plt.yticks([])
plt.show()

