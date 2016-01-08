# encoding: utf-8
import sys
import numpy as np
from skimage import io, transform
from glob import glob
import random

def main():
    if len(sys.argv) < 4:
        print "./crop_negative.py INPUT_DIR OUTPUT_DIR N"
        return
    print "start."
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    n_negatives = int(sys.argv[3])

    image_list = glob('%s/*.jpeg' % input_dir) + glob('%s/*.png' % input_dir)
    count = 0
    for i, image_path in enumerate(image_list):
        image = io.imread(image_path)
        print image.shape
        assert image.shape[0] >= 128 or image.shape[1] >= 128
        while n_negatives * (i + 1) / len(image_list) > count:
            print i
            cropped = crop_randomly(image)
            io.imsave('%s/%d.jpg' % (output_dir, count), cropped)
            count += 1

def crop_randomly(image):
    h, w, _ = image.shape
    x = random.randint(0, w - 128)
    y = random.randint(0, h - 128)
    cropped = image[y:y + 128, x:x + 128]
    return cropped

if __name__ == "__main__":
    main()
