import pylab
import imageio
import skimage
import numpy as np
import caffe
import sys

argvs = sys.argv
argc = len(argvs)

if argc is not 6:
    print '[Usage]python gro_movie.py <movie file path> <network model> <trained model> <mean file> <cpu or gpu>'
    sys.exit()


filename = '/Users/masai/Desktop/company/NTT/movies/ISIS_14/1401.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

imageio

# Neural Network Model
filename = argvs[1]
MODEL_FILE = argvs[2]
PRETRAINED = argvs[3]
MEAN = argvs[4]
MODE = argvs[5]

NUMOFCANDIDATES = 1


# make neural network
if MODE == 'gpu':
    caffe.set_mode_gpu()
elif MODE == 'cpu':
    caffe.set_mode_cpu()
else:
    print 'can not recognize cpu or gpu MODE.'
    exit()

net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN).mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

try:
    for num, image in enumerate(vid.iter_data()):
        if num % int(vid._meta['fps']):
            continue
        else:
            fig = pylab.figure()
            pylab.imshow(image)
            img = skimage.img_as_float(image).astype(np.float32)
            if len(img) <= 145 or len(img) <= 145:
                continue
            #print type(img)
            #print img.shape
            inputs_ = []
            inputs_.append(img)
            try:
                predictions = net.predict(inputs_, oversample=False)
                #print predictions
            except Exception as e:
                print "predict exception:"
                print e
                continue
            # calc score
            predictions = np.sort(predictions, axis=0)
            print predictions
            timestamp = float(num)/ vid.get_meta_data()['fps']
            print(timestamp)
            fig.suptitle('image #{}, timestamp={}, violence_score={}'.format(num, timestamp, predictions[0]), fontsize=20)
        
            pylab.show()
except RuntimeError:
    print('something went wrong')
