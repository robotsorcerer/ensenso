"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.
The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
TODO: implement data_augment for training
Ellis Brown, Max deGroot
"""

# Adapted from Max deGroot's ssd.pytorch code
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/data_augment.py

import cv2
import numpy as np

class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network
    dimension -> tensorize -> color adj
    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize)).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.Tensor(img)
