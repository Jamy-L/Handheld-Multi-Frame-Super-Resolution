# Handheld Multi-Frame Super-Resolution

[[Paper]](https://www.ipol.im/pub/pre/460) [[Demo (to appear)]](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=460)

 This repository contains a non-official implementation of the “Handheld Multi-Frame Super-Resolution algorithm” paper by Wronski et al. (used in the Google Pixel 3 camera), which performs simultaneously multi-image super-resolution demosaicking and denoising from a burst of raw photgraphs. To the best of our knowledge, this is the first publicly available comprehensive implementation of this well-acclaimed paper, for which no official code has been released so far.
 
 The original paper can be found [here](https://sites.google.com/view/handheld-super-res/), whereas our publication detailing the implementation is available on [IPOL](https://www.ipol.im/pub/pre/460). In this companion publication, we fill the implementation blanks of the original SIGGRAPH paper, and disclose many details to actually implement the method.
 Note that our Numba-based implementation is not as fast as that of Google. It is mainly for scientific and educational purpose, with a special care given to make the code as readable and understandable as possible, and was not optimized to minimize the execution time or the memory usage as in an industrial context. Yet, on high-end consumer grade GPUs (NVIDIA RTX 3090 GPU), a 12MP burst of 20 images is expected to generate a 48MP image within less than 4 seconds (without counting Numba's just-in-time compilation), which is enough for running comparisons, or being the base of a faster implementation. 

 We hope this code and the details in the IPOL publication will help the image processing and computational photography communities, and foster new top-of-the-line super-resolution approaches. Please find below two examples of demosaicking and super-resolution from a real raw burst from [this repository](https://github.com/goutamgmb/deep-rep).

![image](https://user-images.githubusercontent.com/46826148/212689891-603e0502-c817-4623-9134-3e7522c72680.png)
![image](https://user-images.githubusercontent.com/46826148/212690127-eb18b00b-5457-44b6-9e27-7f9f88159f4a.png)


## Get started
First of all, install the requirements by running (PyTorch is solely used for the GPU-based FFT as there is none in Numba):
```bash
pip install -r requirements.txt
```

Then, run the approach for x2 super-resolution with:
```python
from handeld_super_resolution import process

# Set the verbose level.
options = {'verbose': 1}

# Specify the scale (1 is demosaicking), the merging kernel, and if you want to postprocess the final image.
params = {
  "scale": 2,
  "merging": {"kernel": "handheld"},
  "post processing": {"on": True}
  }

# Run the algorithm.
burst_path = 'path/to/folder/containing/raw/files'
output_img = process(burst_path, options, params)
```

An example of pipeline call can be found in `example.py`. Simply call `handheld_super_resolution.process(burst_path, options, params)` where :
<ul>
  <li><code>burst_patch</code> is a string containing the file containing .dng files.</li>
  <li><code>options</code> is an optionnal dictionnary containing the verbose option, where higher number means more details during the execution <code>{'verbose' : 1}</code> for example.</li>
  <li><code>params</code> is an optional dictionanry containing all the parameters of the pipleine (such as the upscaling factor). The pipeline is designed to automatically pick some of the parameters based on an estimation of the image SNR and the rest are set to default values, but they can be overwritten by simply assignin a value in <code>params</code>.</li>
</ul>

To obtain the bursts used in the publication, please download the latest release of the repo. It contains the code and two raw bursts of respectively 13 images from [[Bhat et al., ICCV21]](https://arxiv.org/abs/2108.08286) and 20 images from [[Lecouat et al., SIGGRAPH22]](https://arxiv.org/abs/2207.14671). Otherwise specify the path to any burst of raw images, e.g., `*.dng`, `*.ARW` or `*.CR2` for instance.

## Citation
If this code or the implementation details of the companion IPOL publication are of any help, please cite our work:
```BibTex
@article{lafenetre23handheld,
  title={Implementing Handheld Burst Super-resolution},
  author={Lafenetre, Jamy and Facciolo, Gabriele and Eboli, Thomas},
  journal={Image Processing On Line},
  year={2023},
}
```

## Troubleshooting
If you encounter any bug, please open an issue and/or sent an email at jamy.lafenetre@ens-paris-saclay.fr and thomas.eboli@ens-paris-saclay.fr.

## Parameters
The list of all the parameters, their default values and their relation to the SNR can be found in `params.py`. Here are a few of them, grouped by category :

### General parameters
|Parameter|usage|
|--|--|
|scale|The upscaling factor, can be floating but should remain bewteen 1 and 3.|
|Ts|Tile size for the ICA algorithm, and the block matching. Is fixed by the SNR.|
|mode|bayer or grey ; the pipeline can processe grey or color image.|
|debug|If turned on, other debug informations can be returned|

### Block matching
|Parameter|usage|
|--|--|
|factors|list of the downsampling factor to generate the Guassian pyramid|
|tileSize|list of the tileSizes during local search. The last stage should always be Ts !|
|searchRadia|The search radius for each stage|
|distances|L1 or L2; the norm to minimize at each stage|

### ICA
|Parameter|usage|
|--|--|
|kanadeIter|Number of iterations of the ICA algorithm|
|sigma blur|Std of the Gausian filter applied to the grey image before computing gradient. If 0, no filter is applied|

### Robustness
|Parameter|usage|
|--|--|
|on|Whether the robustness is activated or not|
|t||
|s1||
|s2||
|Mt||

### Merging
|Parameter|usage|
|--|--|
|kernel|handheld or iso; whether to use the steerable kernels or the isotropic constant ones (Experiment 3.5in the IPOL article)|
|k_detail||
|k_denoise||
|D_tr||
|D_th||
|k_shrink||
|k_stretch||

### Others
The default floating number representation and the default threads per block number can be modified in <code>utils.py</code>.

### Accumulated robustness denoiser
3 options are available : Gaussian filter during post-processing, median filter during post-processing or filtering during the merge of the reference image (See section 23 of the IPOL article). All of them can be turned on or off and parametrized freely, although the last one gave the best results.

### Known Issues
- The threshold functions and all the hyper-parameters mentionned in the IPOL article have only been partially tweaked : better results are expected with an in depth optimization.
- For images whose estimated SNR is under 14, the tile size should be 64 but the block matching module cannot handle that.
