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

argvs = sys.argv;
argc = len(argvs)

if argc is not 7:
    print '[Usage]python classify_multi_caffe.py <directory> <network model> <trained model> <mean file> <output directory> <skip>'
    sys.exit()

# image directory
directory = argvs[1]

# Neural Network Model
MODEL_FILE = argvs[2]
PRETRAINED = argvs[3]
MEAN = argvs[4]
output_directory = argvs[5]
skip = int(argvs[6])

NUMOFCANDIDATES = 1

# make neural network
caffe.set_mode_gpu();
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN).mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

for i, example in enumerate(iglob('%s/*.jpg' % directory)):
    print("No.%d" % (i))
    if i < skip:
        continue
    sttime = time.time()
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
    inputs_.append(input_image)

    if len(input_image) <= 145 and len(input_image[0]) <= 145:
        continue
    try:
        predictions = net.predict(inputs_, oversample=False)
        #print predictions
    except Exception as e:
        print "predict exception:"
        print e
        continue

    # calc score
    predictions = np.sort(predictions, axis=0)
    top_n = 1
    ero_score = 0
    for i, scores in enumerate(predictions[::-1]):
        if i >= top_n:
            break
        ero_score += scores[0]
    ero_score = ero_score / top_n

    if ero_score > 7.0:
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        shutil.copy(example, output_directory)   

    print('%s\t%f' % (example, ero_score))
    endtime = time.time()
    print "duration: " + str(endtime - sttime)
