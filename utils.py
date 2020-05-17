"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import numpy as np
import copy
from skimage import transform
import imageio

import nibabel as nib

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_test_data(image_path, isCT, iSlice, fine_size0=384, fine_size1=256):

    imgAll = nib.load(image_path)
    img = imgAll.get_data()[int(iSlice), :, :].astype('single')
    if isCT:
        img = (img + 1000.) / (3500. + 1000.) * 255.
    else:
        img = (img - 0.) / (3500. - 0.) * 255.
    img[img > 255.] = 255.

    imgzm = imgAll.get_header().get_zooms()
    imgsz = imgAll.shape
    img_resize_1 = fine_size1
    img_resize_0 = round(imgsz[1] * imgzm[1] * img_resize_1 / (imgsz[2] * imgzm[2]))

    pad_size = int(fine_size0) - img_resize_0
    img = transform.resize(img, (img_resize_0, img_resize_1), preserve_range=True)
    img = np.pad(img, ((int(pad_size // 2), int(pad_size) - int(pad_size // 2)), (0, 0)), mode='constant',
                 constant_values=0)
    # img = np.log(img + 1.)/np.log(2)/4. - 1.
    img = img / 127.5 - 1.
    return img


def load_train_data(batch_file, idA, idB, load_size0=400, load_size1=284, fine_size0=384, fine_size1=256, is_testing=False):

    dataA = r'dataset/CT/InputNorm_id{:0>2d}_slice{:0>3d}_CT.npy'.format(
        idA, int(batch_file[0]))
    dataB = r'dataset/T1/InputNorm_id{:0>2d}_slice{:0>3d}_T1.npy'.format(
        idB, int(batch_file[1]))
    img_A = np.load(dataA)
    img_B = np.load(dataB)

    if not is_testing:

        padA_size0 = load_size0 - img_A.shape[0]
        padA_size1 = load_size1 - img_A.shape[1]
        padB_size0 = load_size0 - img_B.shape[0]
        padB_size1 = load_size1 - img_B.shape[1]

        img_A = np.pad(img_A, ((int(padA_size0 // 2), int(padA_size0) - int(padA_size0 // 2)),
                               (int(padA_size1 // 2), int(padA_size1) - int(padA_size1 // 2))), mode='constant',
                       constant_values=-1)
        img_B = np.pad(img_B, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                               (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                       constant_values=-1)
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size0 - fine_size0)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size1 - fine_size1)))
        img_A = img_A[h1:h1 + fine_size0, w1: w1 + fine_size1]
        img_B = img_B[h1:h1 + fine_size0, w1: w1 + fine_size1]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:

        padA_size = fine_size0 - img_A.shape[0]
        padB_size = fine_size0 - img_B.shape[0]

        img_A = np.pad(img_A, ((int(padA_size // 2), int(padA_size) - int(padA_size // 2)), (0, 0)), mode='constant',
                       constant_values=-1)
        img_B = np.pad(img_B, ((int(padB_size // 2), int(padB_size) - int(padB_size // 2)), (0, 0)), mode='constant',
                       constant_values=-1)

    img_AB = np.dstack((img_A, img_B))
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) * 127.5
