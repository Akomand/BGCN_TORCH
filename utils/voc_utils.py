import os
import random

import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image

"""
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
"""
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def data_partition_random(data, label_n_per_class):
    test_set_n = 1000
    val_set_n = 500

    # THE WHOLE SET
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels, order = load_data(dataset_name)

    N = len(data.y)  # the sample number (LENGTH OF TARGET VECTOR)
    # print(N)
    K = data.y.max() + 1  # the class number
    # print(K)

    labels = data.y.cpu()  # labels vector

    train_index_new = np.zeros(K * label_n_per_class).astype(int)  # total number of training samples (1D vector)
    train_mask_new = np.zeros(N).astype(bool)  # N-dimensional training mask
    val_mask_new = np.zeros(N).astype(bool)  # N-dimensional validation mask
    test_mask_new = np.zeros(N).astype(bool)  # N-dimensional testing mask

    y_train_new = np.zeros((N, K))  # N TARGET VALUES EACH K DIMENSIONAL (NUM CLASSES)
    y_val_new = np.zeros((N, K))  #
    y_test_new = np.zeros((N, K))

    # BUILDING TRAIN, TEST, AND VALIDATION SETS

    # CONSISTS OF A CLASS MAPPED TO ARRAY OF INDICES THAT HAVE SAME CLASS (NODES THAT HAVE SAME CLASS)
    class_index_dict = {}

    for i in range(K):
        # RETURNING THE ARRAY (EACH TIME CALLED) OF INDICES HAVING CLASS i
        # if array([1, 1, 2, 3, 4, 5])
        # return np.where(labels==1)[0] --> array([0, 1])
        class_index_dict[i] = np.where(labels == i)[
            0]  # where() means if condition true pick x value else pick y value. same dim as arr.

    for i in range(K):
        class_index = class_index_dict[i]
        # RANDOMLY SAMPLE 5 (label n per class) INDICES from class i
        train_index_one_class = np.random.choice(class_index, label_n_per_class, replace=False)
        print("The training set index for class {} is {}".format(i, train_index_one_class))

        # SLICES OF 5 (TAKES RANDOMLY SAMPLED INDICES (FROM CLASS i) AND PLACES THEM INTO total training samples vector)
        train_index_new[i * label_n_per_class:(i + 1) * label_n_per_class] = train_index_one_class

    train_index_new = list(train_index_new)
    # print(train_index_new)

    # SET DIFFERENCE (REMOVE THE 5-SAMPLED ARRAY INDICES WE JUST CREATED)
    test_val_potential_index = list(set([i for i in range(N)]) - set(train_index_new))
    # print(len(test_val_potential_index))
    # FROM TEST SET ABOVE, RANDOMY SAMPLE 1000 INDICES
    test_index_new = np.random.choice(test_val_potential_index, test_set_n, replace=False)

    # SET DIFFERENCE (REMOVE THE 1000-SAMPLED TEST ARRAY INDICES WE JUST CREATED)
    potential_val_index = list(set(test_val_potential_index) - set(test_index_new))
    # print(len(potential_val_index))
    # FROM TEST SET ABOVE, RANDOMY SAMPLE 500 INDICES
    val_index_new = np.random.choice(potential_val_index, val_set_n, replace=False)

    # INDICATE WHICH INDICES ARE TRAIN SAMPLES
    train_mask_new[train_index_new] = True
    # INDICATE WHICH INDICES ARE VAL SAMPLES
    val_mask_new[val_index_new] = True
    # INDICATE WHICH INDICES ARE TEST SAMPLES
    test_mask_new[test_index_new] = True

    print(len(train_index_new))
    # Making new one-hot vectors (partition)
    for i in train_index_new:
        y_train_new[i][labels[i]] = 1

    for i in val_index_new:
        y_val_new[i][labels[i]] = 1

    for i in test_index_new:
        y_test_new[i][labels[i]] = 1

    return train_mask_new, val_mask_new, test_mask_new


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode, root):
    assert mode in ['train', 'val', 'validate', 'test', 'inference']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    elif mode == 'val' or mode == 'validate':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    elif mode == 'test' or mode == 'inference':
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    else:
        raise Exception("Please choose proper mode for data")
    return items


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))
