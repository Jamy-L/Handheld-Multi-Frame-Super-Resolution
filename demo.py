import os
import numpy as np
import torch
import argparse
from skimage import img_as_float32, img_as_ubyte, io, color


from handheld_super_resolution import process



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
parser.add_argument('--scale', type=float, default=2, help='Scaling factor')
parser.add_argument('--verbose', type=int, default=1, help='Verbose option (0 to 4)')

## Robustness
parser.add_argument('--t', type=float,  default=0.12, help='Threshold for robustness')
parser.add_argument('--s1', type=float,  default=2, help='Threshold for robustness')
parser.add_argument('--s2', type=float,  default=12, help='Threshold for robustness')
parser.add_argument('--Mt', type=float,  default=0.8, help='Threshold for robustness')
parser.add_argument('--R_on', type=bool,  default=True, help='Whether robustness is activated or not')

parser.add_argument('--R_denoising_on', type=bool, default=True, help='Whether or not the robustness based denoising should be applied')


## Post Processing
parser.add_argument('--post_process', type=bool,  default=True, help='Whether post processing should be applied or not')
parser.add_argument('--do_sharpening', type=bool,  default=True, help='Whether sharpening should be applied during post processing')
parser.add_argument('--radius', type=float,  default=3, help='If sharpening is applied, radius of the unsharp mask')
parser.add_argument('--ammount', type=float,  default=1.5, help='If sharpening is applied, ammount of the unsharp mask')
parser.add_argument('--do_tonemapping', type=bool,  default=True, help='Whether tonnemaping should be applied during post processing')
parser.add_argument('--do_gamma', type=bool,  default=True, help='Whether gamma curve should be applied during post processing')


## Merging (advanced)
parser.add_argument('--kernel_shape', type=str, default='handheld', help='"handheld" or "iso" : Whether to use steerable or isotropic kernels')
parser.add_argument('--k_detail', type=float, default=None, help='SNR based by default')
parser.add_argument('--k_denoise', type=float, default=None, help='SNR based by default')
parser.add_argument('--k_stretch', type=float, default=4)
parser.add_argument('--k_shrink', type=float, default=2)

## Alignment (advanced)
parser.add_argument('--ICA_iter', type=int, default=3, help='Number of ICA Iterations')


args = parser.parse_args()



print('Handheld runs with parameters:')
print('  Upscaling factor:       %d' % args.scale)
if args.R_on:
    print('  Robustness:       enabled')
    print('  -------------------------')
    print('  t:                      %d' % args.t)
    print('  s1:                     %d' % args.s1)
    print('  s2:                     %d' % args.s2)
    print('  Mt:                     %d' % args.Mt)
    if args.R_denoising_on:
        print('  Robustness denoising:   enabled')
    else:
        print('  Robustness denoising:   disabled')
    print('                            ' )
else:
    print('                            ' )

print('  ICA Iterations:         %s' % args.ICA_iter)
print('  Kernel shape:           %s' % args.kernel_shape)
print('  k_stretch:              %d' % args.k_stretch)
print('  k_shrink:               %d' % args.k_shrink)
if args.k_detail is not None:
    print('  k_detail:               %1.2f' % args.k_detail)
else:
    print('  k_detail:               SNR based' )
    
if args.k_denoise is not None:
    print('  k_denoise:              %1.2f' % args.k_denoise)
else:
    print('  k_denoise:              SNR based' )
















options = {'verbose' : args.verbose}
params={"scale": args.scale,
        "merging":{"kernel":args.kernel_shape},
        "robustness":{"on":args.R_on},
        "kanade": {"tuning": {"kanadeIter":args.ICA_iter}}
        }


params['robustness']['tuning'] = {'t' : args.t,
                                  's1' : args.s1,
                                  's2' : args.s2,        
                                  'Mt' : args.Mt,       
                                  }

if args.k_detail is not None:
    params['merging']['tuning']['k_detail'] = args.k_detail
if args.k_denoise is not None:
    params['merging']['tuning']['k_denoise'] = args.k_denoise

params['merging'] = {'tuning' : {'k_stretch' : args.k_stretch,
                                 'k_shrink' : args.k_shrink
                                 }}
params['accumulated robustness denoiser'] = {'on': args.R_denoising_on}

handheld_output = process(args.impath, options, params)



io.imsave(args.outpath, img_as_ubyte(handheld_output))

print('done')