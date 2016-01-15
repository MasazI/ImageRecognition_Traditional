import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import os.path
import caffe
import math
from skimage import data
from skimage import io

from PIL import Image

from image_load import load

import matplotlib.pyplot as plt

import time

from glob import iglob
import shutil

directory = '/Users/masai/Desktop/company/NTT/movies/isis_thumbnails/all_thumb'

for i, example in enumerate(iglob('%s/*.jpg' % directory)):
    print("No.%d" % (i))
    inputs_ = []
    ioimage = None # skimage of original image
 
    try:
        ioimage = io.imread(example)
        print type(ioimage)
        print ioimage.shape
    except:
        print "ioimage error."
        continue
 
    # load caffe
    try:
        input_image = load(ioimage)
        print type(input_image)
        print input_image.shape
    except:
        print "input_image error"
        continue
