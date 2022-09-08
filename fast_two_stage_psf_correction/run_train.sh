## Tiny model
#python training_ca_removal.py --use_tiny --device cuda:1 --batch_size 40 --patch_size 128 --n_epochs 2000 --epoch_to_save 50 \
# --weight_path /mnt/ddisk/teboli/blind_psf/results/ca_removal/

## Tiny model + demosaicing
# python training_ca_removal.py --use_tiny --device cuda:1 --do_demosaicing --batch_size 40 --patch_size 128 --n_epochs 2000 \
# --epoch_to_save 50 --weight_path /mnt/ddisk/teboli/blind_psf/results/ca_removal/

## Tiny model + Gaussian filters
# python training_ca_removal.py --use_tiny --device cuda:1 --use_gaussian_filters --batch_size 40 --patch_size 128 --n_epochs 2000 \
# --epoch_to_save 50 --weight_path /mnt/ddisk/teboli/blind_psf/results/ca_removal/ --num_workers 4


## Tiny model + Gaussian filters + chroma supervision GOOD ONE!
# python training_ca_removal.py --use_tiny --device cuda:1 --use_gaussian_filters --do_chroma --batch_size 40 --patch_size 128 \
# --n_epochs 2000 --epoch_to_save 25 --weight_path /mnt/ddisk/teboli/blind_psf/results/ca_removal/ --num_workers 4


## Super tiny model + Gaussian filters + chroma supervision GOOD ONE!
python train.py --use_super_tiny --device cuda:0 --use_gaussian_filters --do_chroma --batch_size 40 --patch_size 128 \
--n_epochs 2000 --epoch_to_save 25  --num_workers 4

