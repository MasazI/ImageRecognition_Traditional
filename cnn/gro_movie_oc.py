import argparse

import os
import sys

import numpy as np
import cv2
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe

import time

cap = cv2.VideoCapture('/Users/masai/Desktop/company/NTT/movies/ISIS_08/0801.mp4')

parser = argparse.ArgumentParser(
    description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
parser.add_argument('movie', help='Path to movie file')
parser.add_argument('model', help='Path to the pretrained Caffe model')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                    help='Path to the mean file')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Zero-origin GPU ID (nevative value indicates CPU)')
parser.add_argument('--skip', '-s', type=int, default=0, help='The number of skip frames.')
args = parser.parse_args()


print('Loading model file.')
func = caffe.CaffeFunction(args.model)
print('Loaded %s', args.model)
if args.gpu >= 0:
    cuda.init(args.gpu)
    func.to_gpu()

print('Loading mean file.')
mean_image = np.load(args.mean)

print('Loaded %s', args.mean)

def predict(x):
    y, = func(inputs={'data': x}, outputs=['loss3/classifier_g2'], disable=['loss1/ave_pool','loss2/ave_pool'], train=False)
    return y

# get video capture
cap = cv2.VideoCapture(args.movie)
cnt = args.skip
while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        cnt = cnt + 1
        cap.set(1, cnt)
        print 'stop'
        exit()

    # debug
    #print 'image check.'
    #print type(frame)
    #print frame.shape
    #print frame.ndim
    #print frame.shape[2]

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # resize for mean
    resized_img = cv2.resize(frame, (256, 256))
    image = resized_img.transpose(2, 0, 1).astype(np.float32)
    image -= mean_image

    # resize for input
    input_img = cv2.resize(resized_img, (224, 224))
    input_shape_image = input_img.transpose(2, 0, 1).astype(np.float32)    
    x_batch = np.ndarray((1, 3, 224, 224), dtype=np.float32)
    x_batch[0] = input_shape_image
    if args.gpu >= 0:
        x_batch=cuda.to_gpu(x_batch)
    

    sttime = time.time()    
    # variable.
    x = chainer.Variable(x_batch, volatile=True)
    # chain.
    score = predict(x)
    endtime = time.time()
    print("frame cnt: %d, duration: %f[sec]" % (cnt, endtime - sttime))
    
    # debug
    #print type(score)
    #print score.data.shape
    #print score.data[0]

    font = cv2.FONT_HERSHEY_DUPLEX
    text = "violence score: " + str(score.data[0])
    if score.data[0][0] < 9:
        cv2.putText(frame, text, (50, 50), font, 1, (255, 0, 0))
    else:
        cv2.putText(frame, text, (50, 50), font, 1, (0, 0, 255))

    if cv2.waitKey(60) >= 0:
        break

    cv2.imshow('Violence Filter Demo.', frame)

    cnt = cnt + 15
    # CV_CAP_PROP_POS_FRAMES is 1   
    cap.set(1, cnt)


cap.release()
cv2.destroyAllWindows()

