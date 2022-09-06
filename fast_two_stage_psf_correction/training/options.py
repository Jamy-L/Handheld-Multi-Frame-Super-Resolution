from argparse import ArgumentParser


def Options():
    parser = ArgumentParser()

    ## Model
    parser.add_argument('--use_tiny', default=False, action='store_true', help="Use the small network (see training.py)")
    parser.add_argument('--use_super_tiny', default=False, action='store_true', help="Use the super small network (see training.py)")
    parser.add_argument('--device', type=str, default='cuda:0', help="Cuda device is many cards are available")
    parser.add_argument('--state_dict_path', type=str, default='./checkpoints', help="Weights path to load a network")
    parser.add_argument('--load_epoch', type=int, default=0, help="Epoch to load")

    ## Augmentation
    parser.add_argument('--do_demosaicing', default=False, action='store_true', help="Do data augmentation with demoisaicking")
    parser.add_argument('--do_denoising', default=False, action='store_true', help="Do data augmentation with denoising")
    parser.add_argument('--do_gamma_compression', default=False, action='store_true', help="Supervision with gamma compression curve?")
    parser.add_argument('--do_chroma', default=False, action='store_true', help="Use the loss on color residual as described in the paper")

    ## Dataset
    parser.add_argument('--patch_size', type=int, default=96, help="Patch size of the training samples")
    parser.add_argument('--images_folder', type=str, default='/mnt/ddisk/teboli/div2k/', help="Folder containing the images to crop")
    parser.add_argument('--psfs_folder', type=str, default='./psfs/', help="Folder containing the Bauer's PSF")
    parser.add_argument('--use_gaussian_filters', default=False, action='store_true', help="Training with Gaussian filter, no real-world PSFs?")
    parser.add_argument('--simulate_saturation', default=False, action='store_true', help="Simulating saturation in the training samples?")
    parser.add_argument('--load_images', default=False, action='store_true', help="Should we load in memory the images? (Just read the images once and store them in memory")

    ## Dataloader
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number for threads to create the training samples")

    ## Optimizer
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--factor', type=float, default=0.5, help="Learning rate schedule discount rate")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of training epochs")

    ## Save
    parser.add_argument('--epoch_to_save', type=int, default=25, help="Step between two weight saving")
    parser.add_argument('--weight_path', type=str, default='./checkpoints', help="Where to save the weights")

    return parser
