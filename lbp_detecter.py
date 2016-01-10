# enGcoding: utf-8

import sys, os

from skimage import transform
from skimage import feature
from skimage import io
from skimage import color
from glob import iglob

import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math

WINDOW_SIZE = 48
CELL_SIZE = 1
THRESHOLD = 1.5
LBP_POINTS = 8
LBP_RADIUS = 1
 
# セルごとのLBP特徴計算
def get_histogram(image, cell_size, lbp_points, lbp_radius):
    lbp = feature.local_binary_pattern(image, lbp_points, lbp_radius, 'uniform')
    bins = lbp_points + 2
    histogram = np.zeros(shape=(image.shape[0]/cell_size, image.shape[1]/cell_size, bins), dtype=np.int)
    for y in range(0, image.shape[0] - cell_size, cell_size):
        for x in range(0, image.shape[1] - cell_size, cell_size):
            #print('image x: %d, y: %d' % (x, y))
            for dy in range(cell_size):
                for dx in range(cell_size):
                    histogram[y/cell_size, x/cell_size, int(lbp[y + dy, x + dx])] += 1
    return histogram


def get_histogram_feature(lbp):
    bins = LBP_POINTS + 2
    histograms = []
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            histogram = np.zeros(shape = (bins,))
            for dy in range(CELL_SIZE):
                for dx in range(CELL_SIZE):
                    histogram[lbp[y + dy, x + dx]] += 1
            histograms.append(histogram)
    return np.concatenate(histograms)


# directory から取り出して特徴量を計算
def get_features(directory):
    features = []
    for fn in iglob('%s/*.jpg' % directory):
        image = color.rgb2gray(io.imread(fn))
        if image.shape[0] != WINDOW_SIZE or image.shape[1] != WINDOW_SIZE:
            print("error size: %d, %d" % (image.shape[0], image.shape[1]))
            continue
        features.append(get_histogram(image, CELL_SIZE, LBP_POINTS, LBP_RADIUS).reshape(-1))
        # flipした画像を追加
        features.append(get_histogram(np.fliplr(image), CELL_SIZE, LBP_POINTS, LBP_RADIUS).reshape(-1))
    return features


def get_features_(directory):
    features = []
    for fn in iglob('%s/*.jpg' % directory):
        image = color.rgb2gray(io.imread(fn))
        lbp_image = feature.local_binary_pattern(
            image, LBP_POINTS, LBP_RADIUS, 'uniform')
        flip_lbp_image = feature.local_binary_pattern(
            np.fliplr(image), LBP_POINTS, LBP_RADIUS, 'uniform')
        features.append(get_histogram_feature(lbp_image))
        features.append(get_histogram_feature(flip_lbp_image))
    return features


# postive画像とnegative画像から特徴量を収集して保存する
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
    X, y = pickle.load(open(data_path))
    classifier = sklearn.svm.LinearSVC(C=0.0001)
    classifier.fit(X, y)
    pickle.dump(classifier, open(save_path, 'w'))


# evaluation svm model
def eval(model_path, eval_path):
    classifier = pickle.load(open(model_path))
    x, y = pickle.load(open(eval_path))
    y_predict = classifier.predict(x)
    correct = 0
    for i in range(len(y)):
        if y[i] == y_predict[i]: correct += 1
    print('accuracy: %f' % (float(correct)/len(y)))


# search image using pyramid of image
def search(query_image, svm):
    WIDTH, HEIGHT = (WINDOW_SIZE, WINDOW_SIZE)
    detections = []
    scale_factor = 2.0 ** (-1.0/8.0)
    target = color.rgb2gray(io.imread(query_image))
    target_scaled = target + 0
    print target.shape
    for s in range(16):
        histogram = get_histogram(target_scaled, CELL_SIZE, LBP_POINTS, LBP_RADIUS)
        for y in range(0, histogram.shape[0] - HEIGHT/CELL_SIZE):
            for x in range(0, histogram.shape[1] - WIDTH/CELL_SIZE):
                #print('x:%d, y:%d' % (x, y))
                feature = histogram[y:y + HEIGHT / CELL_SIZE, x:x + WIDTH / CELL_SIZE].reshape(-1)
                score = svm.decision_function(feature)
                category = svm.predict(feature)
                if score[0] > THRESHOLD:
                    print('detect! %f', score[0])
                    scale = (scale_factor ** s)
                    detections.append({'x': x * CELL_SIZE/scale, 'y': y * CELL_SIZE/scale, 'width': WIDTH/scale, 'height': HEIGHT/scale, 'score': score[0], 'category': category[0]})
        target_scaled = transform.rescale(target_scaled, scale_factor)
    print('the number of detections: %d' % (len(detections)))
    return detections

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
    if union == 0: return 0
    return intersect / union


def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def test_lbp():
    print '-- test just lbp feature'
    #image1 = color.rgb2gray(io.imread('./flag/positive/posi_ISIS_0101010048_00000015.jpg'))
    #image2 = color.rgb2gray(io.imread('./flag/positive/posi_ISIS_0101010217_00000082.jpg'))
    #image3 = color.rgb2gray(io.imread('./flag/positive/posi_ISIS_0202010084_00000453.jpg'))

    image1 = color.rgb2gray(io.imread('./flag/negative/113.jpg'))
    image2 = color.rgb2gray(io.imread('./flag/negative/256.jpg'))
    image3 = color.rgb2gray(io.imread('./flag/negative/1024.jpg'))
 
 
    lbp1 = feature.local_binary_pattern(image1, LBP_POINTS, LBP_RADIUS, 'uniform')
    lbp2 = feature.local_binary_pattern(image2, LBP_POINTS, LBP_RADIUS, 'uniform')
    lbp3 = feature.local_binary_pattern(image3, LBP_POINTS, LBP_RADIUS, 'uniform')
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                           figsize=(9, 6))
    plt.gray()
    ax1.imshow(image1)
    ax1.axis('off')
    hist(ax4, lbp1)
    ax4.set_ylabel('img1')

    ax2.imshow(image2)
    ax2.axis('off')
    hist(ax5, lbp2)
    ax5.set_ylabel('img2')

    ax3.imshow(image3)
    ax3.axis('off')
    hist(ax6, lbp3)
    ax6.set_ylabel('img2')

    plt.show()


def test():

    print '-- test lbp'
    image = io.imread('images/image1.jpg')
    gray = color.rgb2gray(image)
    histogram = get_histogram(gray, CELL_SIZE, LBP_POINTS, LBP_RADIUS)
    feature_vec = histogram.reshape(-1)
    print len(feature_vec)
    print feature_vec
    print histogram.shape
    print histogram
    print feature_vec

    print '-- test extract train features'
    pos_dir = './flag/positive'
    neg_dir = './flag/negative'
    features_save_path = './examples/flag_v2.pkl'
    get_pos_and_neg(pos_dir, neg_dir, features_save_path)

    print '-- test extract eval features'
    eval_pos_dir = './flag/test/positive'
    eval_neg_dir = './flag/test/negative'
    eval_features_save_path = './examples/flag_v2_eval.pkl'
    get_pos_and_neg(eval_pos_dir, eval_neg_dir, eval_features_save_path)

    print '-- test train'
    train_save_path = './train/train_flag_v2.pkl'
    train(features_save_path, train_save_path)
       
    print '-- test eval accuracy'
    eval(train_save_path, eval_features_save_path)

    print "-- test detection"
    classifier = pickle.load(open(train_save_path))
    print './query/flag_positive_easy_1.jpg'
    detections = search('./query/flag_positive_easy_1.jpg', classifier)

    if len(detections) == 0:
        print "can not detect."
        exit()

    detections = sorted(detections, key = lambda d: d['score'], reverse = True)
    deleted = set()
    
    for i in range(len(detections)):
        if i in deleted: continue
        for j in range(i + 1, len(detections)):
            if nms(detections[i], detections[j]) > 0.3:
                deleted.add(j)
    detections = [d for i, d in enumerate(detections) if not i in deleted]
    print detections
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    query = io.imread('./query/flag_positive_easy_1.jpg')
    ax.imshow(query)
    for detection in detections:
        x = detection['x']
        y = detection['y']
        width = detection['width']
        height = detection['height']
        rect = mpatches.Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()

def test_detect(query_path):
    classifier = pickle.load(open('./train/train_flag_v2.pkl'))
    detections = search(query_path, classifier)
    if len(detections) == 0:
        print "can not detect."
        exit()

    detections = sorted(detections, key = lambda d: d['score'], reverse = True)
    deleted = set()
    print detections
    for i in range(len(detections)):
        if i in deleted: continue
        for j in range(i + 1, len(detections)):
            if nms(detections[i], detections[j]) > 0.3:
                deleted.add(j)
    detections = [d for i, d in enumerate(detections) if not i in deleted]
    print detections
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    query = io.imread(query_path)
    ax.imshow(query)
    for detection in detections:
        x = detection['x']
        y = detection['y']
        width = detection['width']
        height = detection['height']
        rect = mpatches.Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "./lbp_detecter.py <train or detect>"
        exit()
    process_type = sys.argv[1]

    if process_type == 'train':
        test()
    elif process_type == 'detect': 
        test_detect("./query/pos_query_1.jpg")
    elif process_type == 'lbp':
        test_lbp()
    else:
        print "ineffective option."
