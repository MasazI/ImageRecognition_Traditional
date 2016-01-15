# encoding: utf-8

import skimage.io
import numpy as np

def load(ioimage, color=True):
    """
    image load function @datasection
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(ioimage).astype(np.float32)
    print img.ndim
    print img.shape[2]
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
            print('image edit.')
    elif img.shape[2] == 4:
        img = img[:, :, :3]
        print('image edit.')
    return img
