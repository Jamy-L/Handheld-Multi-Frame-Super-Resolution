# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:51:42 2023

@author: jamyl
"""

import os
import glob

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import torch
from skimage import img_as_float32, img_as_ubyte, io, color, filters
import cv2
import rawpy

from handheld_super_resolution import process
from handheld_super_resolution.utils_dng import save_as_dng 

def print_parameters(config):
    print('\nParameters:\n')

    print(f'  Upscaling factor:       {config.scale}\n')
    if config.scale == 1:
        print('    Demosaicking mode')
    else:
        print('    Super-resolution mode.')
        if config.scale > 2:
            print('    WARNING: Since the optics and the integration on the sensor limit the aliasing,')
            print('             do not expect more details than that obtained at x2 (refer to our paper).')

    print()

    if config.robustness.enabled:
        print('  Robustness:             enabled')
        print('  ------------------------------')
        print(f'  t:                      {config.robustness.tuning.t:.2f}')
        print(f'  s1:                     {config.robustness.tuning.s1:.2f}')
        print(f'  s2:                     {config.robustness.tuning.s2:.2f}')
        print(f'  Mt:                     {config.robustness.tuning.Mt:.2f}')
        if any([config.accumulated_robustness_denoiser.median.enabled,
                config.accumulated_robustness_denoiser.gauss.enabled,
                config.accumulated_robustness_denoiser.merge.enabled]):
            print('  Robustness denoising:   enabled')
    else:
        print('  Robustness:             disabled')
    
    print('\n  Alignment:')
    print('  ------------------------------')
    print(f'  ICA Iterations:         {config.ica.tuning.n_iter}')

    print('\n  Fusion:')
    print('  ------------------------------')
    print(f'  Kernel shape:           {config.merging.kernel}')
    print(f'  k_stretch:              {config.merging.tuning.k_stretch:.2f}')
    print(f'  k_shrink:               {config.merging.tuning.k_shrink:.2f}')
    print(f'  k_detail:               {config.merging.tuning.k_detail:.2f}' if not isinstance(config.merging.tuning.k_detail, str) else '  k_detail:               SNR based')
    print(f'  k_denoise:              {config.merging.tuning.k_denoise:.2f}' if not isinstance(config.merging.tuning.k_denoise, str) else '  k_denoise:              SNR based')

    if config.noise_model.alpha is not None:
        print(f'  alpha:                  {config.noise_model.alpha:.2f}')
        print(f'  beta:                   {config.noise_model.beta:.2f}')

    print()

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #### Argparser
    def str2bool(v):
        v = str(v)
    
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    ## Image parameter
    parser.add_argument("--config", type=str, help="Path to custom config YAML")
    parser.add_argument('--impath', type=str, help='Input burst path')
    parser.add_argument('--outpath', type=str, help='Output image path')
    parser.add_argument("overrides", nargs="*", help="Overrides in key=value format, e.g., ica.tuning.n_iter=4")
    
    args = parser.parse_args()

    default_conf = OmegaConf.load("configs/default.yaml")

    # If user provides a config, load and merge it
    if args.config:
        user_conf = OmegaConf.load(args.config)
        config = OmegaConf.merge(default_conf, user_conf)
    else:
        config = default_conf

    # Apply overrides
    for item in args.overrides:
        key, value = item.split("=", 1)
        # Try to parse int/float/bool automatically
        try:
            value = eval(value)
        except:
            raise ValueError(f"Could not parse override value: {value}")
        OmegaConf.update(config, key, value)
    
    print_parameters(config)
    
    #### Handheld ####
    print('Processing with handheld super-resolution')
    if config.noise_model.alpha or config.noise_model.beta:
        assert config.noise_model.beta and config.noise_model.alpha, 'Both alpha and beta should be provided'

    outpath = Path(args.outpath)

    # disabling post processing for dng outputs
    if outpath.suffix == '.dng':
        config.postprocessing.enabled = False
    
    handheld_output, debug_dict = process(args.impath, config)
    handheld_output = np.nan_to_num(handheld_output)
    handheld_output = np.clip(handheld_output, 0, 1)
    
    
    # define a faster imsave for large png images
    def imsave(fname, rgb_8bit_data):
        return cv2.imwrite(fname,  cv2.cvtColor(rgb_8bit_data, cv2.COLOR_RGB2BGR ))
    
    
    #### Save images ####
    
    if outpath.suffix == '.dng':
        if config.verbose >=1 :
            print('Saving output to {}'.format(outpath.with_suffix('.dng').as_posix()))
        ref_img_path = glob.glob(os.path.join(args.impath, '*.dng'))[0]
        save_as_dng(handheld_output, ref_img_path, outpath)
        
    else:
        imsave(args.outpath, img_as_ubyte(handheld_output))

    if config.robustness.save_mask and debug_dict.get('accumulated robustness', None) is not None:
        n_images = len(glob.glob(os.path.join(args.impath, '*.dng')))
        rob = debug_dict['accumulated robustness'].copy_to_host()/(n_images-1)
        rob = np.repeat(rob[..., None], 3, axis=-1)
        # Upscale NN to output scale
        rob = cv2.resize(rob, (handheld_output.shape[1], handheld_output.shape[0]), interpolation=cv2.INTER_NEAREST)
        imsave(Path(args.outpath).with_suffix('.rob.png'), img_as_ubyte(rob))
