import csv

import numpy as np
import tensorflow as tf
import cv2


def check_repeat(seq):
    seen = set()
    seen_add = seen.add
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    return list(seen_twice)


def binarize(sketch, threshold=245):
    sketch[sketch < threshold] = 0
    sketch[sketch >= threshold] = 255
    return sketch


def showImg(img):
    cv2.imshow("test", img)
    cv2.waitKey(-1)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        l = list(reader)

    return l


def read_txt(filename):
    with open(filename) as txtfile:
        lines = txtfile.readlines()
    return [l[:-1] for l in lines]


def split_csvlist(stat_info):
    cat = list(set([item['Category'] for item in stat_info]))
    l = []
    for c in cat:
        li = [item for item in stat_info if item['Category'] == c]
        l.append(li)

    return cat, l
