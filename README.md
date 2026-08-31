# Handheld Multi-Frame Super-Resolution

[[Paper]](https://www.ipol.im/pub/pre/460) [[Demo]](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=460)

**Update 02/23/23: We have now made the demo on the IPOL plateform publicly available. You can use it with you own raw images.**

This repository contains a non-official implementation of the “Handheld Multi-Frame Super-Resolution algorithm” paper by Wronski et al. (used in the Google Pixel 3 camera), which performs simultaneously multi-image super-resolution demosaicking and denoising from a burst of raw photgraphs. To the best of our knowledge, this is the first publicly available comprehensive implementation of this well-acclaimed paper, for which no official code has been released so far.
 
The original paper can be found [here](https://sites.google.com/view/handheld-super-res/), whereas our publication detailing the implementation is available on [IPOL](https://www.ipol.im/pub/pre/460). In this companion publication, we fill the implementation blanks of the original SIGGRAPH paper, and disclose many details to actually implement the method.
Note that our Numba-based implementation is not as fast as that of Google. It is mainly for scientific and educational purpose, with a special care given to make the code as readable and understandable as possible, and was not optimized to minimize the execution time or the memory usage as in an industrial context. Yet, on high-end consumer grade GPUs (NVIDIA RTX 3090 GPU), a 12MP burst of 20 images is expected to generate a 48MP image within less than 4 seconds (without counting Numba's just-in-time compilation), which is enough for running comparisons, or being the base of a faster implementation. 

We hope this code and the details in the IPOL publication will help the image processing and computational photography communities, and foster new top-of-the-line super-resolution approaches. Please find below two examples of demosaicking and super-resolution from a real raw burst from [this repository](https://github.com/goutamgmb/deep-rep). 

![image](https://user-images.githubusercontent.com/46826148/212689891-603e0502-c817-4623-9134-3e7522c72680.png)
![image](https://user-images.githubusercontent.com/46826148/212690127-eb18b00b-5457-44b6-9e27-7f9f88159f4a.png)


#### Post-processing

In the examples above and in our IPOL paper, we used the post-processing approach of this [repo](https://github.com/teboli/fast_two_stage_psf_correction) to remove the remaining optical aberrations on the following examples. 
Check also our publicly available implementation of **Polyblur** in this [repo](https://github.com/teboli/polyblur) to sharpen the result you get with this super-resolution code.

## Get started
>⚠️ For windows users, we recommend to perform the install using WSL to avoid potential issues that can be encountered with numba (see issue [#48](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution/issues/48)).

Start by creating a the following conda environment and activate it. Notice that numba can be tricky to install correctly, and you may need to adjust the `"cuda-version"` part based on your setup.
```bash
conda create -n handheld -y -c conda-forge -c pytorch \
  python=3.9 numpy=1.26.4 \
  numba-cuda "cuda-version=12" \
  rawpy exifread scipy scikit-image opencv colour-demosaicing matplotlib tqdm
conda activate handheld
```
You will need a cuda toolkit. Again, please adapt the cuda version:
```
conda install -c nvidia "cuda-toolkit=12.8"
```
Lastly, install pytorch (you need to ensure that the installed versio support cuda). This is a minimal example but may not work for everyone:
```
pip install torch
```
### Setting noise profile (OPTIONAL)
Note that the latest releases of the code automatically run `fast_monte_carlo.py` everytime you launch the program, which basically computes the noise curves on-the-fly using a fast approximation. You can therefore ignore this section.

We have provided in `./data` the noise curves for the correction of the robustness ratio for the Google Pixel 4a camera by taking the affine noise coefficients from the EXIF tags, and subsequently ran
```bash
python monte_carlo_simulation.py
```
If you want to precompute your own curves taylored to your device, please replace `alpha` and `beta` in this script with the coefficients of your camera, and run the MC simulator to generate the correction curves at several ISO levels tailored for you specific device. Our curves may work for you camera but it might be sub-optimal as the noise models of the Google Pixel 4a camera and yours may diverge.


### Running the code
Place your .dng image burst in the `./test_burst/` folder. You can download some dng bursts [here](https://github.com/goutamgmb/deep-rep), or download the latest release of the code already containing test bursts. Now, simply run the code for x2 super-resolution with:
```
python run_handheld.py --impath test_burst --outpath output.png
```

You can also use the following canvas in your own scripts:
```python
from handheld_super_resolution.config import Config
from handheld_super_resolution.super_resolution import process

# Typed dataclasses contain all defaults.
config = Config(scale=2)

# calling the pipeline
burst_path = './test_burst/Samsung/'
output_img = process(burst_path, config)[0]
```

The core API is `handheld_super_resolution.process(burst_path, config)`, where
`config` is an instance of `handheld_super_resolution.Config`. The function makes
a private runtime copy, so the same configuration object can be reused safely.

To obtain the bursts used in the publication, please download the latest release of the repo. It contains the code and two raw bursts of respectively 13 images from [[Bhat et al., ICCV21]](https://arxiv.org/abs/2108.08286) and 20 images from [[Lecouat et al., SIGGRAPH22]](https://arxiv.org/abs/2207.14671). Otherwise specify the path to any burst of raw images, e.g., `*.dng`, `*.ARW` or `*.CR2` for instance. The result is found in the `./results/` folder. Remember that if you have activated the post-processing flag, the predicted image will be further tone-mapped and sharpened. Deactivate it if you want to plug in your own ISP.

## Generating DNG

Saving images as DNG requires extra-dependencies and steps:
### 1. Install imageio using pip:

    ```bash
    pip install imageio
    ```

### 2. Install exiftool. On linux:
```
sudo apt update
sudo apt install -y libimage-exiftool-perl
exiftool -ver  # Should show version 12.40 or higher
```
You can download the appropriate version for your operating system from the [exiftool website](https://exiftool.org/).

### 3. Compile dng_validate (Adobe DNG SDK)
The `dng_validate` tool is required to finalize DNG files. Adobe provides source code but no longer pre-compiled binaries.
You need build dependencies:
```
sudo apt install -y build-essential gcc g++ make unzip libjpeg-dev
```
We will compile using the community makefile from a`bworrall/go-dng` which provides Linux build support:
```
cd ~/projects  # or your preferred location

# Download DNG SDK with community makefile
git clone https://github.com/abworrall/go-dng.git
cd go-dng/sdk
```
Before compiling, we need to disable JPEG thumbnails, that doesn't work for big images and would prevent from generating them (we get the "No JPEG encoder" error. If you have another solution, please open an issue).
#### Patch the source code
During compilation, the normal process would be to unzip `dng_sdk_1_6.zip` and compile. We need to manually unzip, change a line in the source, then compile with the modified source.
```
cd go-dng/sdk
unzip dng_sdk_1_6.zip
cd dng_sdk/source

# Edit dng_validate.cpp line 401
# Change: for (uint32 previewIndex = 0; previewIndex < 2; previewIndex++)
# To:     for (uint32 previewIndex = 0; previewIndex < 0; previewIndex++)

# Using sed:
sed -i 's/previewIndex < 2/previewIndex < 0/' dng_validate.cpp

cd ../../
```

We can now compile

```
# Compile dng_validate
make CC=g++ CXX=g++ # Use system compiler, and not conda's (to use apt installed packages)
# Say NO when prompted to replace the dezipped files
# Verify compilation
ls -lh bin/dng_validate  # Should show ~25MB binary
./bin/dng_validate        # Should show usage information
```

That's it! Once you've completed these steps you should be able to save the output as DNG, for example by running:
```
python run_handheld.py --impath test_burst --outpath output.dng
```
## About the noise profile
The method requires the noise profile of the camera, which consists of $\alpha$ and $\beta$, two parameters that depends on the camera and its settings. In a nutshell, they describe how the variance of the noise $\sigma_n^2$ evolves with the brightness $I$ : $\sigma_n^2 = \alpha\times I + \beta$. They are necessary to compute the noise curves in `monte_carlo_simulation.py` or `fast_monte_carlo.py`, and to perform the generalised Anscombe transform (GAT) in `utils_image.py`.

By default, the program reads the `noise profile` tag of the dng stack to determine $\alpha$ and $\beta$. If these values are unavailable or wrong, you can provide your own values of $\alpha$ and $\beta$. If the stack was captured using a smartphone, the noise profile of the Pixel 4 can give good results : 
```python
    alpha = 1.80710882e-4 * ISO / 100
    beta = 3.1937599182128e-6 * (ISO / 100)**2
```

For better results, determining the accurate noise profile of your device can be done using tools [such as this demo](https://www.ipol.im/pub/art/2013/45/)

## Citation
If this code or the implementation details of the companion IPOL publication are of any help, please cite our work:
```BibTex
@article{lafenetre23handheld,
  title={Implementing Handheld Burst Super-Resolution},
  author={Lafenetre, Jamy and Facciolo, Gabriele and Eboli, Thomas},
  journal={Image Processing On Line},
  year={2023},
}
```

## Parameters
All settings are typed dataclass fields in `handheld_super_resolution.config`.
The command line is generated from those types with tyro:

```bash
python run_handheld.py --impath test_burst --outpath output.png \
  --scale 2 --merging.kernel iso \
  --noise-model.alpha 0.0000123 --noise-model.beta 0.000001
```

Boolean options use tyro's paired flags. For example, disable post-processing
with `--postprocessing.no-enabled`. Lists are space-separated:

```bash
python run_handheld.py --impath test_burst --outpath output.png \
  --block-matching.tuning.factors 1 2 4 4
```

Run `python run_handheld.py --help` for the complete configuration reference,
including defaults and allowed choices. YAML configuration and positional
`key=value` overrides are no longer supported.

## Troubleshooting
If you encounter any bug, please open an issue and/or sent an email at jamy.lafenetre@ens-paris-saclay.fr and thomas.eboli@ens-paris-saclay.fr.


### Others
The default floating number representation and the default threads per block number can be modified in <code>utils.py</code>.

### Accumulated robustness denoiser
3 options are available : Gaussian filter during post-processing, median filter during post-processing or filtering during the merge of the reference image (See section 23 of the IPOL article). All of them can be turned on or off and parametrized freely, although the last one gave the best results.

### Known Issues
- The threshold functions and all the hyper-parameters mentionned in the IPOL article have only been partially tweaked : better results are expected with an in depth optimization.
