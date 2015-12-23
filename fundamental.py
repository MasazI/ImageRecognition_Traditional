#encoding: utf-8

from skimage import feature
from skimage import io
from skimage import color

import pickle
import numpy as np

image = io.imread('images/image1.jpg')

print('1: %s' % (type(image)))
print('2: %s' % (image.shape,))
print('3: %s' % (image[300,400],))

# rgb = [0, 0, 250]
image[180, 240, 0:3] = [0, 0, 250]
image[20:140, 20:200, 0:3] = [0, 0, 0]
image[200:300, 50:430, 0:3] -= 30


# lbp features
LBP_POINTS = 24
LBP_RADIUS = 3
gray = color.rgb2gray(image)
feature.local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, 'uniform')


# セルごとのLBP特徴計算
def get_histogram(image, cell_size, lbp_points, lbp_radius):
    lbp = feature.local_binary_pattern(gray, lbp_points, lbp_radius, 'uniform')
    bins = lbp_points + 2
    histogram = np.zeros(shape=(image.shape[0]/cell_size, image.shape[1]/cell_size, bins), dtype=np.int)

    for y in range(0, image.shape[0] - cell_size, cell_size):
        for x in range(0, image.shape[1] - cell_size, cell_size):
            for dy in range(cell_size):
                for dx in range(cell_size):
                    histogram[y/cell_size, x/cell_size, int(lbp[y + dy, x + dx])] += 1
    return histogram

histogram = get_histogram(gray, 4, LBP_POINTS, LBP_RADIUS)
feature_vec = histogram.reshape(-1)

print histogram.shape
print histogram
print feature_vec


# directory から取り出して特徴量を計算
def get_features(directory):
    features = []
    for fn in iglob('%s/*.jpg' % directory):
        image = color.rgb2gray(io.imread(fn))
        features.append(get_histogram(image).reshape(-1))
        # flipした画像を追加
        features.append(get_histogram(np.fliplr(image)).reshape(-1))

def get_pos_and_neg(pos_dir, neg_dir, save_path):
    positive_samples = get_features(pos_dir)
    negative_samples = get_features(neg_dir)

    n_pos = len(positive_samples)
    n_neg = len(negative_samples)

    x = np.array(positive_samples + negative_samples)
    y = np.array([1 for i in range(n_pos)] + [0 for i in range(n_neg)])

    pickle.dump((x, y), open(save_path, 'w'))


# train svm model
import sklearn.svm
def train(data_path, save_path):
    x, y = pickle.load(open(data_path))
    classifier = sklearn.svm.LinearSVC(c = 0.0001)
    classifier.fit(x, y)
    picle.dump(classifier, open(save_path, 'w'))


# evaluation svm model
def eval(model_path, eval_path):
    classifier = pickle.load(open(model_path))
    x, y = pickle.load(open(eval_path))
    y_predict = classifier.predict(x)
    correct = 0
    for i in range(len(y)):
        if y[i] == y_predict[i]: correct += 1
    print('accuracy: %f' % (float(correct)/len(y)))


# non-maximum suppression
def nms(a, b):
    '''
    args: [x, y, width, height, score]
    '''
    left = max(a['x'], b['x'])
    right = min(a['x'] + a['width'], b['x'] + b['width'])
    top = max(a['y'], b['y'])
    bottom = min(a['y'] + a['height'], b['y'] + b['height'])
    intersect = max(0, (right - left) * (bottom - top))
    union = a['width'] * a['height'] + b['width'] * b['height'] - intersect
    return intersection / union

# scoreでソート
detections = sorted(detections, key = lambda d: d['score'], reverse = True)

deleted = set()

for i in range(len(detections)):
    if i in deleted : continue
    for j in range(i + 1, len(detections)):
        if nms(detections[i], detections[j]) > 0.3:
            deleted.add(j)
detections = [d for i, d in enumerate(detections) if not i in deleted]



io.imshow(image)
io.show()
