# enGcoding: utf-8

from skimage import feature
from skimage import io
from skimage import color
from glob import iglob

import pickle
import numpy as np


# セルごとのLBP特徴計算
def get_histogram(image, cell_size, lbp_points, lbp_radius):
    lbp = feature.local_binary_pattern(image, lbp_points, lbp_radius, 'uniform')
    bins = lbp_points + 2
    histogram = np.zeros(shape=(image.shape[0]/cell_size, image.shape[1]/cell_size, bins), dtype=np.int)

    for y in range(0, image.shape[0] - cell_size, cell_size):
        for x in range(0, image.shape[1] - cell_size, cell_size):
            for dy in range(cell_size):
                for dx in range(cell_size):
                    histogram[y/cell_size, x/cell_size, int(lbp[y + dy, x + dx])] += 1
    return histogram


# directory から取り出して特徴量を計算
def get_features(directory):
    LBP_POINTS = 24
    LBP_RADIUS = 3
    features = []
    for fn in iglob('%s/*.jpg' % directory):
        image = color.rgb2gray(io.imread(fn))
        features.append(get_histogram(image, 4, LBP_POINTS, LBP_RADIUS).reshape(-1))
        # flipした画像を追加
        features.append(get_histogram(np.fliplr(image), 4, LBP_POINTS, LBP_RADIUS).reshape(-1))
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
    print len(X)
    print len(y)
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


# スコアの計算
def compute_score_map(template, target):
    th, tw = template.shape
    score_map = np.zeros(shape=(target.shape[0] - th, target.shape[1] - tw))
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            diff = target[y:y + th, x:x + tw] - template
            score_map[y, x] = np.square(diff).sum()
    return score_map


# search image using pyramid of image
def search(query_image, svm):
    WIDTH, HEIGHT = (128, 128)
    CELL_SIZE = 4
    THRESHOLD = 3
    detections = []
    scale_factor = 2.0 ** (-1.0/8.0)
    target = color.rgb2gray(io.imread(query_image))
    target_scaled = target + 0
    for s in range(16):
        histogram = get_histogram(target_scaled)
        for y in range(0, histogram.shape[0] - HEIGHT / CELL_SIZE):
            for x in range(0, histogram.shape[1] - WIDTH / CELL_SIZE):
                feature = histogram[y:y + HEIGHT / CELL_SIZE, x:x + WIDTH / CELL_SIZE].reshape(-1)
                score = svm.decision_function(feature)
                if score[0] > THRESHOLD:
                    scale = (scale_facgor ** s)
                    detections.append({'x:' x * cell_size/scale, 'y:' y * cell_size/scale, 'width': WIDTH/scale, 'height': HEIGHT/scale, 'score': score})
        target_scaled = transform.rescale(target_scaled, scale_factor)


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
    return intersect / union


def test():
    print '-- test lbp'
    LBP_POINTS = 24
    LBP_RADIUS = 3
    image = io.imread('images/image1.jpg')
    gray = color.rgb2gray(image)
    histogram = get_histogram(gray, 4, LBP_POINTS, LBP_RADIUS)
    feature_vec = histogram.reshape(-1)
    print histogram.shape
    print histogram
    print feature_vec

    print '-- test extract train features'
    pos_dir = './flag/positive'
    neg_dir = './flag/negative'
    features_save_path = './examples/flag_v1.pkl'
    get_pos_and_neg(pos_dir, neg_dir, features_save_path)

    print '-- test extract eval features'
    eval_post_dir = './flag/test/positive'
    eval_neg_dir = './flag/test/negative'
    eval_features_save_path = './examples/flag_v1_eval.pkl'
    get_pos_and_neg(eval_post_dir, eval_neg_dir, eval_features_save_path)

    print '-- test train'
    train_save_path = './train/train_flag_v1.pkl'
    train(features_save_path, train_save_path)
       
    print '-- test eval accuracy'
    eval(train_save_path, eval_features_save_path)
    


def detect():
    # scoreでソート
    detections = sorted(detections, key = lambda d: d['score'], reverse = True)
    deleted = set()
    
    for i in range(len(detections)):
        if i in deleted : continue
        for j in range(i + 1, len(detections)):
            if nms(detections[i], detections[j]) > 0.3:
                deleted.add(j)
    detections = [d for i, d in enumerate(detections) if not i in deleted]


if __name__ == '__main__':
    test()

#io.imshow(image)
#io.show()
