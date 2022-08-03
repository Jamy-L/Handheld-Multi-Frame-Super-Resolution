# hdrplus-python
Open source Python implementation of the HDR+ photography pipeline, originally developped by Google and presented in a [2016 article](https://dl.acm.org/doi/pdf/10.1145/2980179.2980254). The finishing pipeline is simplified and of lesser quality than the one described in the original publication.

For an interactive demo and the associated article, _**An Analysis and Implementation of the HDR+ Burst Denoising Method**_, check out [Image Processing On Line](https://www.ipol.im/pub/art/2021/336/)

_Note: A C++ / Halide implementation by different authors is available [here](https://github.com/timothybrooks/hdr-plus/)._

## Installation Instructions
All the libraries necessary to run the code are listed in the hdrplus.yml Conda environment file.
Simply run
```
conda env create -f hdrplus.yml
```
from a command window to install a functional environment.


## File Contents and Provided Files
All source code containing algorithm functions is located within the `package/algorithm` folder,
except some optional visualization functions located in `package/visualization/vis.py`

Scripts to run the algorithm are located at the root of the repo.

## Running the Code
Two scripts are provided to either run the algorithm on a single burst (`runHdrplus.py`)
or on a series of bursts all within the same parent folder (`runHdrplus_multiple.py`).

Examples of use:
```
python runHdrplus.py -i ./test_data/33TJ_20150606_224837_294 -o ./results_test1 -m full -v 2
```
```
python runHdrplus_multiple.py -i ./test_data -o ./results_test2 -m full
```
	
You can run the algorithm in three modes (`-m` command argument):
- **full**:
	- required inputs (per burst folder): all raw .dng burst files and a single reference_frame.txt file
	- outputs (per burst folder): 3 .jpg images: final image `X_final.jpg` + minimally processed versions of the reference and merged image `X_reference_gamma.jpg` `X_merged_gamma.jpg`
- **align**: 
	- required inputs: all raw .dng burst files and a single reference_frame.txt files
	- outputs: a .dng file (copy of the reference image) + 2 numpy files: `X_aligned_tiles.npy` and `X_padding.npy`
- **merge**:
	- required inputs: (obtained from align mode) a single .dng file (for metadata of the reference image) + 2 numpy files `X_aligned_tiles.npy` and `X_padding.npy`
	- outputs: a .dng file (copy of the reference image) + 1 numpy file: `X_merged_bayer.npy`
- **finish**:
	- required inputs: (obtained from merge mode) a single .dng file (for metadata of the reference image) + 1 numpy file (for actual pixel values) `X_merged_bayer.npy`
	- outputs: final image `X_final.jpg`
You can also change the values of the `'write___'` dictionary items in `params.py` to change the kind of files dumped in each mode (at your own risk).

A helper script for the minimal processing of raw .dng files into .png/.jpg files (e.g. for the visualization of input images) is also included in the code: `all_dngs_to_png.py`

## Test Data
1 burst can be found in the `test_data` folder (each burst being in its own subfolder)
Feel free to add your own data. The structure of a burst folder must be the following:
- the burst name is specified by the name of the folder itself
- burst images must be stored as .dng files (most proprietary raw images formats can be turned to DNG using [Adobe DNG Converter](https://helpx.adobe.com/photoshop/using/adobe-dng-converter.html))
- image files must be named the following way: `commonpart<X>.dng`, where `<X>` gives an indication of the frame number (eg `payload_N000.dng`, `payload_N001.dng` / `G0140178.dng`, `G0140179.dng`)
- you can specify the reference frame by putting a zero-indexed number inside a `reference_frame.txt` file (i.e. 0 for the 1st frame)

Additional data can be downloaded via the following links:
- gopro bursts created for the purpose of the IPOL article: https://drive.google.com/drive/folders/1j2NIEPSnrdjS0sjL1kzl3VEmYKBkChJD?usp=sharing
- HDR+ dataset from the original article: https://hdrplusdata.org/dataset.html
- curated subset of bursts used in the IPOL demo: https://drive.google.com/drive/folders/1bHttOqV_R7QLJPLkLVlffRInZFlNKB1q?usp=sharing

## Citation

If you find this work useful in your research or publication, please cite our paper:
```
@article{ipol.2021.336,
    title   = {{An Analysis and Implementation of the HDR+ Burst Denoising Method}},
    author  = {Monod, Antoine and Delon, Julie and Veit, Thomas},
    journal = {{Image Processing On Line}},
    volume  = {11},
    pages   = {142--169},
    year    = {2021},
    note    = {\url{https://doi.org/10.5201/ipol.2021.336}}
}
```

## COPYRIGHT AND LICENSE INFORMATION
Copyright (c) 2021 Antoine Monod

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

This file implements an algorithm possibly linked to the patent US9077913B2.
This file is made available for the exclusive aim of serving as scientific tool
to verify the soundness and completeness of the algorithm description.
Compilation, execution and redistribution of this file may violate patents rights in certain countries.
The situation being different for every country and changing over time,
it is your responsibility to determine which patent rights restrictions apply to you
before you compile, use, modify, or redistribute this file.
A patent lawyer is qualified to make this determination.
If and only if they don't conflict with any patent terms,
you can benefit from the following license terms attached to this file.
