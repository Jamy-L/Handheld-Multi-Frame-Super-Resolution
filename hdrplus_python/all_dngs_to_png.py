import os
import glob
import rawpy
import cv2

if __name__ == "__main__":
	burstFolder = r'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus-python/test_data/33TJ_20150606_224837_294'
	outputFolder = r'C:/Users/jamyl/Documents/GitHub/Handheld-Multi-Frame-Super-Resolution/hdrplus-python/Original'
	extension = '.png'  # replace this to '.jpg' if preferred
	rawpyArgs = {
		'demosaic_algorithm' : rawpy.DemosaicAlgorithm.DHT,  # replace by rawpy.DemosaicAlgorithm.LINEAR for bilinear interpolation
		'half_size' : False,
		'use_camera_wb' : True,  # apply white balance with coefficients saved at capture time
		'use_auto_wb' : False,
		'no_auto_bright': True,  # set this to False if you want image brightness to be automatically changed (not recommended)
		'output_color' : rawpy.ColorSpace.sRGB,
		'gamma': (2.4, 12.92),  # approximates the sRGB gamma compression function
		'output_bps' : 8
	}

	# Get the list of all .dng images in the burst path
	rawPathList = glob.glob(os.path.join(burstFolder, '*.dng'))
	# sort the raw list in ascending order to avoid errors in reference frame selection
	rawPathList.sort()
	assert rawPathList != [], 'At least one raw .dng file must be present in the burst folder.'
	# get a minimally processed 8 bit sRGB version of each .dng image
	for i, rawPath in enumerate(rawPathList):
		print("processing image {}/{}: {}".format(i + 1, len(rawPathList), rawPath))
		with rawpy.imread(rawPath) as rawObject:
			# get the processed 8bit sRGB matrix
			outputImage = rawObject.postprocess(**rawpyArgs)
		outputName = os.path.join(outputFolder, os.path.basename(rawPath)[:-4] + extension)
		# use whatever library you want (cv2, imageio, pillow/PIL) to save the image
		if extension == '.png':
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))
		elif extension == '.jpg':
			cv2.imwrite(outputName, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
