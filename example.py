# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:10:18 2023


@author: jamyl
"""


import matplotlib.pyplot as plt
from handheld_super_resolution import process


# Specify verbose options
options = {'verbose' : 1}



# Specify the scale as follows. All the parameters are automatically 
# choosen but can be overwritten : check params.py to see the entire list
# of configurable parameters.
params={
        "scale":2,
        "merging" : {
            'kernel': 'handheld'},
        'post processing' : {'on':True}
        # Post processing is enabled by default,
        # but it can be turned off here
        
        }

# calling the pipeline
burst_path = 'test_burst/Samsung/'
output_img = process(burst_path, options, params)


# plotting the result
plt.figure("output")
plt.imshow(output_img, interpolation = 'none')
plt.xticks([])
plt.yticks([])