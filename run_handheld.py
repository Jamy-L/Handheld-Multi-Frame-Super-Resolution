# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:51:42 2023

@author: jamyl
"""

import os
import glob

import argparse
from pathlib import Path
import numpy as np
import torch
from skimage import img_as_float32, img_as_ubyte, io, color, filters
import cv2
import rawpy

from handheld_super_resolution import process
from handheld_super_resolution.utils_dng import save_as_dng 

def print_parameters(args):
    print('\nParameters:\n')

    print(f'  Upscaling factor:       {args.scale}\n')
    if args.scale == 1:
        print('    Demosaicking mode')
    else:
        print('    Super-resolution mode.')
        if args.scale > 2:
            print('    WARNING: Since the optics and the integration on the sensor limit the aliasing,')
            print('             do not expect more details than that obtained at x2 (refer to our paper).')

    print()
    
    if args.R_on:
        print('  Robustness:             enabled')
        print('  ------------------------------')
        print(f'  t:                      {args.t:.2f}')
        print(f'  s1:                     {args.s1:.2f}')
        print(f'  s2:                     {args.s2:.2f}')
        print(f'  Mt:                     {args.Mt:.2f}')
        print(f'  Robustness denoising:   {"enabled" if args.R_denoising_on else "disabled"}')
    else:
        print('  Robustness:             disabled')
    
    print('\n  Alignment:')
    print('  ------------------------------')
    print(f'  ICA Iterations:         {args.ICA_iter}')

    print('\n  Fusion:')
    print('  ------------------------------')
    print(f'  Kernel shape:           {args.kernel_shape}')
    print(f'  k_stretch:              {args.k_stretch:.2f}')
    print(f'  k_shrink:               {args.k_shrink:.2f}')
    print(f'  k_detail:               {args.k_detail:.2f}' if args.k_detail is not None else '  k_detail:               SNR based')
    print(f'  k_denoise:              {args.k_denoise:.2f}' if args.k_denoise is not None else '  k_denoise:              SNR based')
    
    if args.alpha is not None:
        print(f'  alpha:                  {args.alpha:.2f}')
        print(f'  beta:                   {args.beta:.2f}')
    
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
    parser.add_argument('--impath', type=str, help='input image')
    parser.add_argument('--outpath', type=str, help='out image')
    parser.add_argument('--scale', type=int, default=2, help='Scaling factor')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose option (0 to 4)')
    parser.add_argument('--save_rob', type=str2bool, default=False, help='Save the accumulated robustness mask')

    ## Override metadata noise profile
    parser.add_argument('--alpha', type=float, default=None, help='alpha value (Noise profile)')
    parser.add_argument('--beta', type=float, default=None, help='beta value (Noise profile)')
    
    ## Robustness
    parser.add_argument('--t', type=float,  default=0.12, help='Threshold for robustness')
    parser.add_argument('--s1', type=float,  default=2, help='Threshold for robustness')
    parser.add_argument('--s2', type=float,  default=12, help='Threshold for robustness')
    parser.add_argument('--Mt', type=float,  default=0.8, help='Threshold for robustness')
    parser.add_argument('--R_on', type=str2bool,  default=True, help='Whether robustness is activated or not')
    
    parser.add_argument('--R_denoising_on', type=str2bool, default=True, help='Whether or not the robustness based denoising should be applied')
    
    
    ## Post Processing
    parser.add_argument('--post_process', type=str2bool,  default=True, help='Whether post processing should be applied or not')
    parser.add_argument('--do_sharpening', type=str2bool,  default=True, help='Whether sharpening should be applied during post processing')
    parser.add_argument('--radius', type=float,  default=3, help='If sharpening is applied, radius of the unsharp mask')
    parser.add_argument('--amount', type=float,  default=1.5, help='If sharpening is applied, amount of the unsharp mask')
    parser.add_argument('--do_tonemapping', type=str2bool,  default=True, help='Whether tonnemaping should be applied during post processing')
    parser.add_argument('--do_gamma', type=str2bool,  default=True, help='Whether gamma curve should be applied during post processing')
    parser.add_argument('--do_color_correction', type=str2bool,  default=True, help='Whether color correction should be applied during post processing')
    
    
    ## Merging (advanced)
    parser.add_argument('--kernel_shape', type=str, default='handheld', help='"handheld" or "iso" : Whether to use steerable or isotropic kernels')
    parser.add_argument('--k_detail', type=float, default=None, help='SNR based by default')
    parser.add_argument('--k_denoise', type=float, default=None, help='SNR based by default')
    parser.add_argument('--k_stretch', type=float, default=4)
    parser.add_argument('--k_shrink', type=float, default=2)
    
    ## Alignment (advanced)
    parser.add_argument('--ICA_iter', type=int, default=3, help='Number of ICA Iterations')
    
    
    args = parser.parse_args()
    
    
    print_parameters(args)    
    
    
    #### Handheld ####
    print('Processing with handheld super-resolution')
    options = {'verbose' : args.verbose}
    params={"scale": args.scale,
            "merging":{"kernel":args.kernel_shape},
            "robustness":{"on":args.R_on},
            "kanade": {"tuning": {"kanadeIter":args.ICA_iter}}
            }
    
    if args.alpha or args.beta:
        assert args.beta and args.alpha, 'Both alpha and beta should be provided'
        params['alpha'] = args.alpha
        params['beta'] = args.beta
    
    
    params['robustness']['tuning'] = {'t' : args.t,
                                      's1' : args.s1,
                                      's2' : args.s2,        
                                      'Mt' : args.Mt,       
                                      }
   
    params['merging'] = {'tuning' : {'k_stretch' : args.k_stretch,
                                     'k_shrink' : args.k_shrink
                                     }}

    if args.k_detail is not None:
        params['merging']['tuning']['k_detail'] = args.k_detail
    if args.k_denoise is not None:
        params['merging']['tuning']['k_denoise'] = args.k_denoise
    
    params['accumulated robustness denoiser'] = {'on': args.R_denoising_on}
    
    outpath = Path(args.outpath)
    # disabling post processing for dng outputs
    if outpath.suffix == '.dng':
        args.post_process = False
    
    
    params['post processing'] = {'on':args.post_process,
                        'do sharpening' : args.do_sharpening,
                        'do tonemapping':args.do_tonemapping,
                        'do gamma' : args.do_gamma,
                        'do devignette' : False,
                        'do color correction': args.do_color_correction,
            
                        'sharpening' : {'radius': args.radius,
                                        'ammount': args.amount}
                        }
    
    handheld_output, debug_dict = process(args.impath, options, params)
    handheld_output = np.nan_to_num(handheld_output)
    handheld_output = np.clip(handheld_output, 0, 1)
    
    
    # define a faster imsave for large png images
    def imsave(fname, rgb_8bit_data):
        return cv2.imwrite(fname,  cv2.cvtColor(rgb_8bit_data, cv2.COLOR_RGB2BGR ))
    
    
    #### Save images ####
    
    if outpath.suffix == '.dng':
        if options['verbose'] >=1 :
            print('Saving output to {}'.format(outpath.with_suffix('.dng').as_posix()))
        ref_img_path = glob.glob(os.path.join(args.impath, '*.dng'))[0]
        save_as_dng(handheld_output, ref_img_path, outpath)
        
    else:
        imsave(args.outpath, img_as_ubyte(handheld_output))
    
    if args.save_rob and debug_dict.get('accumulated robustness', None) is not None:
        n_images = len(glob.glob(os.path.join(args.impath, '*.dng')))
        rob = debug_dict['accumulated robustness'].copy_to_host()/(n_images-1)
        rob = np.repeat(rob[..., None], 3, axis=-1)
        # Upscale NN to output scale
        rob = cv2.resize(rob, (handheld_output.shape[1], handheld_output.shape[0]), interpolation=cv2.INTER_NEAREST)
        imsave(Path(args.outpath).with_suffix('.rob.png'), img_as_ubyte(rob))
