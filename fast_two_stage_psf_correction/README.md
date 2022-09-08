# Fast two-step blind optical aberration correction

This repository contains the official implemention of the ECCV'22 paper *Fast two-step blind 
optical aberration correction* (paper <a href="https://arxiv.org/abs/2208.00950">here</a>).

### Testing

First install the requirements with
> pip install -r requirements.txt

You can run the testing script with
> python test.py

where you can change the parameters and the image path to test your own example.

### Package installation

If you want to deploy this code in another project, you can install the package with
> python setup.py install

You can import the package in your python code with
> import fast_optics_correction

When done, you can call the main module in *modules.py* as 
> fast_optics_correction.OpticsCorrection

Please refer to *test.py* for an example.

### Training

Download the DIV2K dataset and the PSFs at
> https://edmond.mpdl.mpg.de/file.xhtml?fileId=101784&version=1.0

and run
> bash prepare_psfs.sh

You are now all set to train the model! You can run the following bash file with preselected options
> bash run_train.sh


### Troubleshooting

In case of questions or bugs, please contact me at <thomas.eboli@ens-paris-saclay.fr>.
