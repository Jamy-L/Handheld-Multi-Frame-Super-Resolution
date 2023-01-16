# Handheld Multi-Frame Super-Resolution
 This repository contains a non official implementation of the method “Handheld Multi-Frame Super-Resolution algorithm” by Wronski et al. (used in the Google Pixel 3 camera), which performs simultaneously multi-image super-resolution demosaicking and denoising from a burst of images.
 
 The original paper and materials can be found [here](https://sites.google.com/view/handheld-super-res/), whereas our publication detailing the implementation is available [on the IPOL website](https://www.ipol.im/). TODO add the link
 It only serves a scientific and educational purpose, and was not optimized to minimize the execution time or the memory usage. On high-end consumer grade GPUs, a 12MP burst of 20 images is expected to generate a 48MP image within less than 4 seconds (without counting Numba's just-in-time compilation).

![image](https://user-images.githubusercontent.com/46826148/212689891-603e0502-c817-4623-9134-3e7522c72680.png)
![image](https://user-images.githubusercontent.com/46826148/212690127-eb18b00b-5457-44b6-9e27-7f9f88159f4a.png)


## Requirements
- Numba with a working CUDA environment (it may be possible to use [the numba CUDA simulator](https://numba.readthedocs.io/en/stable/cuda/simulator.html) if you have no GPU, but it has not been tested).
- pyTorch with a working CUDA environment.
- [exifread](https://pypi.org/project/ExifRead/) and [rawpy](https://pypi.org/project/rawpy/) to read .dng files.
- Noise profile curves. The curves included in the repo are for the camera of the Google Pixel 4a. They may be used for other cameras, but generating the adapted noise model using <code>monte_carlo_simulation.py()</code> would be preferable.
- An NVIDIA GPU with a [compute capability](https://developer.nvidia.com/cuda-gpus) of at least 2.0. This condition may be removed by optimizing further the block matching module.


## Calling the pipeline
An example of pipeline call can be found in `example.py`. Simply call `handheld_super_resolution.process(burst_path, options, params)` where :
<ul>
  <li><code>burst_patch</code> is a string containing the file containing .dng files.</li>
  <li><code>options</code> is an optionnal dictionnary containing the verbose option, where higher number means more details during the execution <code>{'verbose' : 1}</code> for example.</li>
  <li><code>params</code> is an optional dictionanry containing all the parameters of the pipleine (such as the upscaling factor). The pipeline is designed to automatically pick some of the parameters based on an estimation of the image SNR and the rest are set to default values, but they can be overwritten by simply assignin a value in <code>params</code>.</li>
</ul>

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
