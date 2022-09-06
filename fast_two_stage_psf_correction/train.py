import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import tqdm

from fast_optics_correction.polyblur_functions import mild_inverse_rank3, blur_estimation
from fast_optics_correction.modules import ResUNet
from fast_optics_correction import utils

from training.datasets import DatasetGaussianBlur
from training.options import Options
from training.metrics import L1WithGamma
from training.demosaicing import HamiltonAdam



def train(model, loader, optimizer, criterion, demosaicker, args):
    l1_loss = 0
    count = 0
    model.train()
    for i, data in tqdm.tqdm(enumerate(loader)):
        # print('    TR: Batch %03d' % (i+1))
        ## Read data
        u, v, k = data
        u = u.to(args.device)
        v = v.to(args.device)
        k = k.to(args.device)

        ## (Optional) Demosaicking augmentation
        if demosaicker is not None:
            if v.shape[1] == 3:
                v = utils.mosaic(v)
            with torch.no_grad():
                v = demosaicker(v).clamp(0.0, 1.0)

        ## Sharpening
        v = mild_inverse_rank3(v, k, correlate=True, halo_removal=False)

        ## Inference
        optimizer.zero_grad()
        v_red_green = torch.cat([v[:, 0:1], v[:, 1:2]], dim=1)  # Bx2xHxW
        v_blue_green = torch.cat([v[:, 2:3], v[:, 1:2]], dim=1)  # Bx2xHxW
        v_red_blue = torch.cat([v[:, 0:1], v[:, 2:3]], dim=0)  # (2B)x1xHxW
        v_red_green_blue_green = torch.cat([v_red_green, v_blue_green], dim=0)  # (2B)x2xHxW
        u_red_blue_pred = v_red_blue - model(v_red_green_blue_green)  # (2B)x1xHxW
        u_red_blue = torch.cat([u[:, 0:1], u[:, 2:3]], dim=0)  # (2B)x1xHxW

        ## Update
        if args.do_chroma:
            u_green_pred = torch.cat([v[:, 1:2], v[:, 1:2]], dim=0)  # (2B)x2xHxW
            u_green = torch.cat([u[:, 1:2], u[:, 1:2]], dim=0)  # (2B)x2xHxW
            error = criterion(u_red_blue_pred - u_green_pred, u_red_blue - u_green)
        else:
            error = criterion(u_red_blue_pred, u_red_blue)
        error.backward()
        optimizer.step()

        ## Record
        l1_loss += error.item()
        count += 1

    l1_loss /= count
    return l1_loss


def validate(model, loader, criterion, demosaicker, args):
    l1_loss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader)):
            # print('    VA: Batch %03d' % (i + 1))
            ## Read data
            u, v, k = data
            u = u.to(args.device)
            v = v.to(args.device)
            k = k.to(args.device)

            ## (Optional) Demosaicking augmentation
            if demosaicker is not None:
                if v.shape[1] == 3:
                    v = utils.mosaic(v)
                with torch.no_grad():
                    v = demosaicker(v).clamp(0.0, 1.0)

            ## Sharpening
            v = mild_inverse_rank3(v, k, correlate=True, halo_removal=False)

            ## Inference - Red and green
            v = v.detach()
            v_red_green = torch.cat([v[:, 0:1], v[:, 1:2]], dim=1)  # Bx2xHxW
            v_blue_green = torch.cat([v[:, 2:3], v[:, 1:2]], dim=1)  # Bx2xHxW
            v_red_blue = torch.cat([v[:, 0:1], v[:, 2:3]], dim=0)  # (2B)x1xHxW
            v_red_green_blue_green = torch.cat([v_red_green, v_blue_green], dim=0)  # (2B)x2xHxW
            u_red_blue_pred = v_red_blue - model(v_red_green_blue_green)  # (2B)x1xHxW
            u_red_blue = torch.cat([u[:, 0:1], u[:, 2:3]], dim=0)  # (2B)x1xHxW

            ## Evaluation
            if args.do_chroma:
                u_green_pred = torch.cat([v[:, 1:2], v[:, 1:2]], dim=0)  # (2B)x2xHxW
                u_green = torch.cat([u[:, 1:2], u[:, 1:2]], dim=0)  # (2B)x2xHxW
                error = criterion(u_red_blue_pred - u_green_pred, u_red_blue - u_green)
            else:
                error = criterion(u_red_blue_pred, u_red_blue)

            l1_loss += error.item()
            count += 1

    l1_loss /= count
    return l1_loss


def main(args):
    ## Set model
    if args.use_tiny:
        model = ResUNet(nc=[16, 32, 64, 64], in_nc=2, out_nc=1)
        model_type = 'tiny'
    elif args.use_super_tiny:
        model = ResUNet(nc=[16, 16, 32, 32], in_nc=2, out_nc=1)
        model_type = 'super_tiny'
    else:
        model = ResUNet(nc=[64, 128, 256, 512], in_nc=2, out_nc=1)
        model_type= 'regular'
    if args.load_epoch > 0:
        state_dict_path = os.path.join(args.state_dict_path, model_type + '_epoch_%04d.pt' % args.load_epoch)
        print('Loading', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)

    ## Set optimizer and loss
    optimizer = Adam(model.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, min_lr=args.min_lr, verbose=True)

    ## Set criterion
    if args.do_gamma_compression:
        criterion = L1WithGamma()
    else:
        criterion = nn.L1Loss()

    ## Go to GPU
    model = model.to(args.device)
    criterion = criterion.to(args.device)

    ## (Optional) If demosaicking augmentation, load a demosaicker
    ## We have chosen Hamilton-Adam since it is fast and good enough
    ## but you can select any Pytorch demosaicker or joint 
    ## denoisier/demosaicker instead, e.g., a CNN.
    if args.do_demosaicing:
        pattern = 'rggb'
        demosaicker = HamiltonAdam(pattern).to(device)
    else:
        demosaicker = None

    ## Set datasets
    dataset_tr = DatasetGaussianBlur(args, training=True, use_gaussian_filters=args.use_gaussian_filters)
    loader_tr = DataLoader(dataset_tr, shuffle=True, batch_size=args.batch_size, drop_last=True,
                           num_workers=args.num_workers)
    dataset_va = DatasetGaussianBlur(args, training=False, use_gaussian_filters=args.use_gaussian_filters)
    loader_va = DataLoader(dataset_va, shuffle=False, batch_size=args.batch_size, drop_last=True,
                           num_workers=args.num_workers)

    l1_tr_all = np.zeros(args.n_epochs)
    l1_va_all = np.zeros(args.n_epochs)
    print('Main training loop')
    for epoch in range(max(0, args.load_epoch), args.n_epochs):
        ## training
        l1_tr = train(model, loader_tr, optimizer, criterion, demosaicker, args)

        ## validation
        l1_va = validate(model, loader_va, criterion, demosaicker, args)
        scheduler.step(l1_tr)

        ## record
        l1_tr_all[epoch] = l1_tr
        l1_va_all[epoch] = l1_va
        print('Epoch %04d: TR = %2.5f | VA = %2.5f' % (epoch+1, l1_tr, l1_va))

        ## Save weights
        if (epoch+1) % args.epoch_to_save == 0:
            weight_path = args.weight_path
            os.makedirs(weight_path, exist_ok=True)
            weight_path = os.path.join(weight_path, model_type + '_epoch_%04d.pt' % (epoch+1))
            torch.save(model.state_dict(), weight_path)

    print('Press any key to finish')
    input()


if __name__ == '__main__':
    opts = Options()
    main(opts.parse_args())
