# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:34:02 2022

@author: jamyl
"""
import os
import glob

# import logging

from tqdm import tqdm
import numpy as np
from math import modf
import cv2
import torch
from numba import vectorize, guvectorize, uint8, uint16, float32, float64, jit, njit, cuda, int32
import exifread
import rawpy
import matplotlib.pyplot as plt
import colour_demosaicing
# colour_demosaicing.demosaicing_CFA_Bayer_Menon2007
# pip install colour-demosaicing

# import demosaicnet
from handheld_super_resolution import process, raw2rgb, get_params

from plot_flow import flow2img
from handheld_super_resolution.utils import crop



def flat(x):
    return 1-np.exp(-x/1000)


def cfa_to_grayscale(raw_img):
    return (raw_img[..., 0::2, 0::2] + 
            raw_img[..., 1::2, 0::2] + 
            raw_img[..., 0::2, 1::2] + 
            raw_img[..., 1::2, 1::2])/4


# logger = logging.getLogger()
# logger.setLevel(logging.WARNING)

# crop_str = "[1638:2600, 1912:2938]" # for friant
# crop_str = "[1002:1686, 2406:3130]" # for rue4 (arrondissement)
# crop_str = "[1002:1686, 2000:2700]" # for rue4 (truck plate)
# crop_str = "[500:2500, 1000:2500]" # for samsung 0
crop_str = None
# crop_str = "[1500:2500, 2000:3000]" # for samsung 1

#%%

params = get_params(PSNR = 35)

# Overwritting default parameters
params["scale"] = 1
options = {'verbose' : 4}

params['merging']['kernel'] = 'handheld'
params['robustness']['on'] = False
params['kanade']['tuning']['kanadeIter'] = 3
burst_path = 'P:/inriadataset/inriadataset/pixel4a/friant/raw/'
# burst_path = 'P:/inriadataset/inriadataset/pixel3a/rue4/raw'
# burst_path = 'P:/0001/Samsung'

params['kanade']['tuning']['sigma blur'] = 1
output_img = process(burst_path, options, params, crop_str)


#%% extracting images locally for comparison 
raw_path_list = glob.glob(os.path.join(burst_path, '*.dng'))
first_image_path = raw_path_list[0]

raw_ref_img = rawpy.imread(first_image_path)

exif_tags = open(first_image_path, 'rb')
tags = exifread.process_file(exif_tags)

ref_img = raw_ref_img.raw_image.copy()
if crop_str is not None : 
    ref_img = crop(ref_img, crop_str, axis=(0,1))

xyz2cam = raw2rgb.get_xyz2cam_from_exif(first_image_path)

comp_images = rawpy.imread(
    raw_path_list[1]).raw_image.copy()[None]
for i in range(2,len(raw_path_list)):
    comp_images = np.append(comp_images, rawpy.imread(raw_path_list[i]
                                                      ).raw_image.copy()[None], axis=0)
if crop_str is not None :
    comp_images = crop(comp_images, crop_str, axis=(1,2))

white_level = tags['Image Tag 0xC61D'].values[0] # there is only one white level

black_levels = tags['Image BlackLevel'] # This tag is a fraction for some reason. It seems that black levels are all integers anyway
black_levels = np.array([int(x.decimal()) for x in black_levels.values])

white_balance = raw_ref_img.camera_whitebalance

CFA = tags['Image CFAPattern']
CFA = np.array([x for x in CFA.values]).reshape(2,2)

ref_img = ref_img.astype(np.float32)
comp_images = comp_images.astype(np.float32)
for i in range(2):
    for j in range(2):
        channel = channel = CFA[i, j]
        ref_img[i::2, j::2] = (ref_img[i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
        ref_img[i::2, j::2] *= white_balance[channel]
        
        comp_images[:, i::2, j::2] = (comp_images[:, i::2, j::2] - black_levels[channel]) / (white_level - black_levels[channel])
        comp_images[:, i::2, j::2] *= white_balance[channel]



ref_img = np.clip(ref_img, 0.0, 1.0)
comp_images = np.clip(comp_images, 0.0, 1.0)


#%% extracting handhled's output data

imsize = output_img.shape

print('Nan detected in output: ', np.sum(np.isnan(output_img)))
print('Inf detected in output: ', np.sum(np.isinf(output_img)))

plt.figure("output, kernel {}, {} LK".format(params['merging']['kernel'], params['kanade']['grey method']))
postprocessed_output = raw2rgb.postprocess(raw_ref_img, output_img, xyz2cam=xyz2cam) 
plt.imshow(postprocessed_output)


colors = "RGB"
bayer = params["merging"]['exif']['CFA Pattern']
pattern = colors[bayer[0,0]] + colors[bayer[0,1]] +colors[bayer[1,0]]+colors[bayer[1,1]]


base = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(ref_img, pattern=pattern)  # pattern is [[G,R], [B,G]] for the Samsung G8


plt.figure("original bicubic")
postprocessed_bicubic = cv2.resize(raw2rgb.postprocess(raw_ref_img, base, xyz2cam=xyz2cam), None, fx = params["merging"]['scale'], fy = params["merging"]['scale'], interpolation=cv2.INTER_CUBIC)
plt.imshow(postprocessed_bicubic)

# demosaicnet_bayer = demosaicnet.BayerDemosaick()
# mosaicnet_output = demosaicnet_bayer(torch.from_numpy(raw_ref_img.raw_image).flip(-1).unsqueeze(0)).squeeze(0).cpu().numpy()
# mosaicnet_output = np.clip(mosaicnet_output, 0, 1).transpose(1,2,0).astype(np.float64).fliplr()

# plt.figure('mosaicnet output')
# plt.figure(mosaicnet_output)

#%% ploting images in crops
crops_x = 2
crops_y = 2
crop_size = (int(output_img.shape[0]/crops_y), int(output_img.shape[1]/crops_x))

for i in range(crops_y):
    for j in range(crops_x):
        plt.figure("output handheld {} {}".format(i,j))
        plt.imshow(postprocessed_output[i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]])
        
        plt.figure("original bicubic {} {}".format(i,j))
        plt.imshow(postprocessed_bicubic[i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]])



#%% Anisotropy and D
D = covs[:, :, :, 0, 0]
A = covs[:, :, :, 0, 1]
l1 = covs[:, :, :, 1, 0]
im_id = 0


plt.figure("D {} (bayer grad)".format(im_id))
plt.imshow(D[im_id], cmap="gray", vmin=0, vmax = 1)
plt.colorbar()

plt.figure("A {}".format(im_id))
plt.imshow(A[im_id], cmap="gray", vmin=1, vmax = 2)
plt.colorbar()

plt.figure("L1 {}".format(im_id))
plt.imshow(l1[im_id], cmap="gray", vmin=0, vmax = 0.01)
plt.colorbar()


#%% eighen vectors
e1 = covs[0,:,:,:,0]
e2 = covs[0,:,:,:,1]

grey_ref_img = (ref_img[::2,::2,] + ref_img[1::2, ::2] + ref_img[::2, 1::2] + ref_img[1::2, 1::2])/4
imsize = grey_ref_img.shape

# quivers for eighenvectors
# Lower res because pyplot's quiver is really not made for that (=slow)
plt.figure('quiver')
scale = 5*1e1
downscale_coef = 4
ix, iy = 0, 0
patchx, patchy = int(imsize[1]/downscale_coef), int(imsize[0]/downscale_coef)
plt.imshow(grey_ref_img[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1)], cmap="gray")

# minus sign because pyplot takes y axis growing towards top. but we need the opposite
plt.quiver(e1[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 0],
            -e1[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 1],width=0.001,linewidth=0.0001, scale=scale)
plt.quiver(e2[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 0],
            -e2[iy*patchy:patchy*(iy+1), ix*patchx:patchx*(ix + 1), 1], width=0.001,linewidth=0.0001, scale=scale, color='b')



#%%
x = upscaled_al[:,:,:,0]%2
x = x.reshape(x.size)
x_sample = np.random.choice(x, size = 1000, replace = False)

y = upscaled_al[:,:,:,1]%2
y = y.reshape(y.size)
y_sample = np.random.choice(y, size = 1000, replace = False)

ds = pd.DataFrame()
ds['x'] = x_sample
ds['y'] = y_sample

sns.pairplot(ds, kind="hist")


#%% warp optical flow

plt.figure('ref grey')    
ref_grey_image = cfa_to_grayscale(ref_img)
plt.imshow(ref_grey_image, cmap= 'gray')
#%%
plt.figure('im 1')    
ref_grey_image = cfa_to_grayscale(comp_images[1])
plt.imshow(ref_grey_image, cmap= 'gray')

#%%
comp_grey_images = cfa_to_grayscale(comp_images)
grey_al = alignment.copy()
grey_al[:,:,:,-2:]*=0.5 # pure translation are downscaled from bayer to grey 

upscaled_grey_al = upscale_alignement(grey_al, ref_grey_image.shape[:2], 16) 
for image_index in range(comp_images.shape[0]):
    warped = warp_flow(comp_grey_images[image_index], upscaled_grey_al[image_index], rgb = False)
    plt.figure("image {}".format(image_index))
    # plt.imsave('P:/images_test/image_{:02d}.png'.format(image_index),warped, cmap = 'gray')
    plt.imshow(warped, cmap = 'gray')
    plt.figure("EQ {}".format(image_index))
    plt.imshow(np.log10((ref_grey_image - warped)**2),vmin = -6, vmax =0 , cmap="gray")
    plt.colorbar()
    
    print("Im {}, EQM = {}".format(image_index, np.mean((ref_grey_image - warped)**2)))

#%% plot flow
maxrad = 10

for image_index in range(comp_images.shape[0]):
    flow_image = flow2img(upscaled_al[image_index], maxrad)
    plt.figure("flow {}".format(image_index))
    plt.imshow(flow_image)

# Making color wheel
X = np.linspace(-maxrad, maxrad, 1000)
Y = np.linspace(-maxrad, maxrad, 1000)
Xm, Ym = np.meshgrid(X, Y)
Z = np.stack((Xm, Ym)).transpose(1,2,0)
# Z = np.stack((-Z[:,:,1], -Z[:,:,0]), axis=-1)
flow_image = flow2img(Z, maxrad)
plt.figure("wheel")
plt.imshow(flow_image)



#%% warp optical flow rgb

plt.figure('ref rgb')    
plt.imshow(raw2rgb.postprocess(raw_ref_img, colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(ref_img, pattern = "GRBG"), xyz2cam=xyz2cam))  

upscaled_al = upscale_alignement(alignment, ref_img.shape, 16*2) 
for image_index in range(comp_images.shape[0]):
    warped = warp_flow(colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(comp_images[image_index], pattern = "GRBG"), upscaled_al[image_index], rgb = True)
    plt.figure("image {} warped".format(image_index))
    plt.imshow(raw2rgb.postprocess(raw_ref_img, warped, xyz2cam=xyz2cam))
for image_index in range(comp_images.shape[0]):
    plt.figure("image {}".format(image_index))
    plt.imshow(raw2rgb.postprocess(raw_ref_img, colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(comp_images[image_index], pattern = "GRBG"), xyz2cam=xyz2cam))
  

#%% histograms

plt.figure("R histogram")
plt.hist(R.reshape(R.size), bins=25)

plt.figure("r histogram")
plt.hist(r.reshape(r.size), bins=25)

X = np.linspace(0, 5, 100)
Y1 = params['robustness']["tuning"]["s1"]*np.exp(-X) - params['robustness']["tuning"]["t"]
Y2 = params['robustness']["tuning"]["s2"]*np.exp(-X) - params['robustness']["tuning"]["t"]
plt.figure("robustness")
plt.plot(X, np.clip(Y1, 0,1), label = "s1 (discontinuous alignment)")
plt.plot(X, np.clip(Y2, 0,1), label = "s2 (continuous alignment)")
plt.xlabel("(d/sigma)²")
plt.ylabel("R")
plt.legend()
#%% kernels
def plot_local_flow(pos, upscaled_flow):
    
    scale = 1
    plt.figure("flow in (x = {}, y = {})".format(pos[1],pos[0]))
    for image_id in range(upscaled_flow.shape[0]):
        plt.arrow(0, 0,
                  upscaled_flow[image_id, pos[0], pos[1], 0],
                  upscaled_flow[image_id, pos[0], pos[1], 1],
                  width=0.01)
    plt.gca().invert_yaxis()
        
    
    
    

def plot_merge(covs, Dist, pos):
    reframed_posx, x = modf(pos[1]/(2*params['scale'])) # these positions are between 0 and 1
    reframed_posy, y = modf(pos[0]/(2*params['scale']))
    x=int(x); y=int(y)
    
    int_cov = (covs[0, y, x]*(1 - reframed_posx)*(1 - reframed_posy) +
               covs[0, y, x+1]*(reframed_posx)*(1 - reframed_posy) + 
               covs[0, y+1, x]*(1 - reframed_posx)*(reframed_posy) + 
               covs[0, y+1, y+1]*reframed_posx*reframed_posy)
    
    cov_i = np.linalg.inv(int_cov*4*params['scale']**2)
    print(int_cov.shape)
    

    
    
    L = np.linspace(-5, 5, 100)
    Xm, Ym = np.meshgrid(L,L)
    Z = np.empty_like(Xm)
    Z = cov_i[0,0] * Xm**2 + (cov_i[1, 0] + cov_i[0, 1])*Xm*Ym + cov_i[1, 1]*Ym**2
    plt.figure("merge in (x = {}, y = {})".format(pos[1],pos[0]))
    plt.pcolor(Xm, Ym, np.exp(-Z/2), vmin = 0, vmax=1)
    plt.gca().invert_yaxis()
    plt.scatter([0], [0], c='k', marker ='o')
    
    
    
    r_index_x = np.where(bayer ==0)[0][0]
    r_index_y = np.where(bayer ==0)[1][0]
    
    b_index_x = np.where(bayer ==2)[0][0]
    b_index_y = np.where(bayer ==2)[1][0]
    
    colors = ['r', 'g', 'b']
    
    
    for image in range(Dist.shape[0]):
        D = Dist[(image,) + pos]
        dist = abs(D[0,1] - D[1,1]) # for scale
        
        
        if D[0,2] != 1 :
            y = 0
            color = D[0, 2]
        else :
            y = 1
            color = D[1, 2]
        
        if color == 0:
            bayer_index_x = r_index_x
            bayer_index_y = r_index_y
        else :
            bayer_index_x = b_index_x
            bayer_index_y = b_index_y
        
        for idx in range(-1,2):
            for idy in range(-1, 2):
                x = D[0,1] + idx*dist
                y = D[0,0] + idy*dist
                
                bayer_idx = (bayer_index_x + idx)%2
                bayer_idy = (bayer_index_y + idy)%2
                c = colors[CFA[bayer_idy, bayer_idx]]
                d = np.array([x, y])
                plt.scatter(x,y,c=c,marker='x')            

    

    plt.colorbar()
    
    

#%% robustness

# for im_id in range(R.shape[0]):
#     plt.figure('r '+str(im_id))
#     plt.imshow(r[im_id], vmax=1, vmin = 0, cmap = "gray", interpolation='none')
    
# for im_id in range(R.shape[0]):
#     plt.figure('R '+str(im_id))
#     plt.imshow(R[im_id], vmax=1, vmin = 0, cmap = 'gray', interpolation='none')
#     plt.colorbar()
    

plt.figure('accumulated r')
plt.imshow(np.sum(r, axis = 0)/r.shape[0], vmax=1, vmin = 0, cmap = "gray", interpolation='none')



# for im_id in range(R.shape[0]):
#     plt.figure('patchwise error '+str(im_id))
#     plt.imshow(R[im_id, 875:1100, 1100:1300, 0], vmax = np.max(R[:, 875:1100, 1100:1300, 0]), cmap="gray", interpolation='none')
#     plt.colorbar()

# for im_id in range(R.shape[0]):
#     plt.figure('std '+str(im_id))
#     plt.imshow(R[im_id, 875:1100, 1100:1300, 1], vmin = 0, vmax = np.max(R[:, 875:1100, 1100:1300, 1]), cmap="gray", interpolation='none')
#     plt.colorbar()

# for im_id in range(R.shape[0]):
#     plt.figure(''+str(im_id))
#     plt.imshow(R[im_id, :, :], vmin = 0, vmax=6, interpolation='none')
#     plt.colorbar()



#%% D curve

D_tr = params["merging"]['tuning']['D_tr']
D_th = params["merging"]['tuning']['D_th']

L = np.linspace(0, 1.5*(D_tr + D_tr*D_th)**2, 300)
Y = np.clip(1-np.sqrt(L)/D_tr + D_th, 0, 1)
plt.figure('D curve')
plt.plot(L, Y)
plt.xlabel('Lambda 1')
plt.ylabel('D')
