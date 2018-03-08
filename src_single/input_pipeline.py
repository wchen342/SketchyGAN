import os
import cv2
import numpy as np
import tensorflow as tf
from data_processing.tfrecord import *

from scipy import ndimage
from config import Config


# TODO Change to Dataset API
sketchy_dir = '../training_data/sketchy'
flickr_dir = '../training_data/flickr_output'


paired_filenames_1 = [os.path.join(sketchy_dir, f) for f in os.listdir(sketchy_dir)
                      if os.path.isfile(os.path.join(sketchy_dir, f))]
paired_filenames_2 = [os.path.join(flickr_dir, f) for f in os.listdir(flickr_dir)
                      if os.path.isfile(os.path.join(flickr_dir, f))]

print("paired file sketchy num: %d" % len(paired_filenames_1))
print("paired file flickr num: %d" % len(paired_filenames_2))

# build class map
class_mapping = []
classes_info = './data_processing/classes.csv'
classes = read_csv(classes_info)
classes_id = [item['Name'] for item in classes]
for name in paired_filenames_1:
    name = os.path.splitext(os.path.split(name)[1])[0].split('_coco_')[0]
    class_id = classes_id.index(name)
    if class_id not in class_mapping:
        class_mapping.append(class_id)
class_mapping = sorted(class_mapping)
for name in paired_filenames_2:
    name = os.path.splitext(os.path.split(name)[1])[0].split('_coco_')[0]
    class_id = classes_id.index(name)
    if class_id not in class_mapping:
        print(name)
        raise RuntimeError
num_classes = len(class_mapping)


def get_num_classes():
    return num_classes


def one_hot_to_dense(labels):
    # Assume on value is 1
    batch_size = int(labels.get_shape()[0])
    return tf.reshape(tf.where(tf.equal(labels, 1))[:, 1], (batch_size,))


def map_class_id_to_labels(batch_class_id, class_mapping=class_mapping):
    batch_class_id_backup = tf.identity(batch_class_id)

    for i in range(num_classes):
        comparison = tf.equal(batch_class_id_backup, tf.constant(class_mapping[i], dtype=tf.int64))
        batch_class_id = tf.where(comparison, tf.ones_like(batch_class_id) * i, batch_class_id)
    ret_tensor = tf.squeeze(tf.one_hot(tf.cast(batch_class_id, dtype=tf.int32), num_classes,
                                       on_value=1, off_value=0, axis=1))
    return ret_tensor


def binarize(sketch, threshold=250):
    return tf.where(sketch < threshold, x=tf.zeros_like(sketch), y=tf.ones_like(sketch) * 255.)


# SKETCH_CHANNEL = 3
SIZE = {True: (64, 64),
        False: (256, 256)}


# Distance map first, then resize
def get_paired_input(paired_filenames, test_mode, distance_map=True, img_dim=(256, 256),
                     fancy_upscaling=False, data_format='NCHW'):
    if test_mode:
        num_epochs = 1
        shuffle = False
    else:
        num_epochs = None
        shuffle = True
    filename_queue = tf.train.string_input_producer(
        paired_filenames, capacity=512, shuffle=shuffle, num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'ImageNetID': tf.FixedLenFeature([], tf.string),
            'SketchID': tf.FixedLenFeature([], tf.int64),
            'Category': tf.FixedLenFeature([], tf.string),
            'CategoryID': tf.FixedLenFeature([], tf.int64),
            'Difficulty': tf.FixedLenFeature([], tf.int64),
            'Stroke_Count': tf.FixedLenFeature([], tf.int64),
            'WrongPose': tf.FixedLenFeature([], tf.int64),
            'Context': tf.FixedLenFeature([], tf.int64),
            'Ambiguous': tf.FixedLenFeature([], tf.int64),
            'Error': tf.FixedLenFeature([], tf.int64),
            'class_id': tf.FixedLenFeature([], tf.int64),
            'is_test': tf.FixedLenFeature([], tf.int64),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
            'image_small_jpeg': tf.FixedLenFeature([], tf.string),
            'sketch_png': tf.FixedLenFeature([], tf.string),
            'sketch_small_png': tf.FixedLenFeature([], tf.string),
            'dist_map_png': tf.FixedLenFeature([], tf.string),
            'dist_map_small_png': tf.FixedLenFeature([], tf.string),
        }
    )

    if img_dim[0] < 64:
        image = tf.image.decode_jpeg(features['image_small_jpeg'], fancy_upscaling=fancy_upscaling)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, (64, 64, 3))
    else:
        image = tf.image.decode_jpeg(features['image_jpeg'], fancy_upscaling=fancy_upscaling)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, (256, 256, 3))

    if img_dim[0] < 64:
        if Config.pre_calculated_dist_map:
            sketch = tf.image.decode_png(features['dist_map_small_png'], channels=3) if distance_map \
                else tf.image.decode_png(features['sketch_small_png'], channels=3)
        else:
            sketch = tf.image.decode_png(features['sketch_small_png'], channels=3)
        sketch = tf.cast(sketch, tf.float32)
        sketch = tf.reshape(sketch, (64, 64, 3))
    else:
        if Config.pre_calculated_dist_map:
            sketch = tf.image.decode_png(features['dist_map_png'], channels=3) if distance_map \
                else tf.image.decode_png(features['sketch_png'], channels=3)
        else:
            sketch = tf.image.decode_png(features['sketch_png'], channels=3)
        sketch = tf.cast(sketch, tf.float32)
        sketch = tf.reshape(sketch, (256, 256, 3))

    # Distance map
    if not Config.pre_calculated_dist_map and distance_map:
        # Binarize
        sketch = binarize(sketch)
        sketch_shape = sketch.shape

        sketch = tf.py_func(lambda x: ndimage.distance_transform_edt(x).astype(np.float32),
                            [sketch], tf.float32, stateful=False)
        sketch = tf.reshape(sketch, sketch_shape)
        # Normalize
        sketch = sketch / tf.reduce_max(sketch) * 255.

    # Resize
    if img_dim[0] != 256:
        image = tf.image.resize_images(image, img_dim, method=tf.image.ResizeMethod.BILINEAR)
        sketch = tf.image.resize_images(sketch, img_dim, method=tf.image.ResizeMethod.BILINEAR)
    # if img_dim[0] > 256:
    #     image = tf.image.resize_images(image, img_dim, method=tf.image.ResizeMethod.BILINEAR)
    #     sketch = tf.image.resize_images(sketch, img_dim, method=tf.image.ResizeMethod.BILINEAR)
    # elif img_dim[0] < 256:
    #     image = tf.image.resize_images(image, img_dim, method=tf.image.ResizeMethod.AREA)
    #     sketch = tf.image.resize_images(sketch, img_dim, method=tf.image.ResizeMethod.AREA)

    # Augmentation
    # Image
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image_large = tf.image.random_hue(image_large, max_delta=0.05)

    # Normalization
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1)
    image += tf.random_uniform(shape=image.shape, minval=0., maxval=1. / 256)  # dequantize
    sketch = sketch / 255.

    image = image * 2. - 1
    sketch = sketch * 2. - 1

    # Transpose for data format
    if data_format == 'NCHW':
        image = tf.transpose(image, [2, 0, 1])
        sketch = tf.transpose(sketch, [2, 0, 1])

    # Attributes
    category = features['Category']
    imagenet_id = features['ImageNetID']
    sketch_id = features['SketchID']
    class_id = features['class_id']
    is_test = features['is_test']
    WrongPose = features['WrongPose']
    Context = features['Context']
    Ambiguous = features['Ambiguous']
    Error = features['Error']

    if not test_mode:
        is_valid = WrongPose + Context + Ambiguous + Error + is_test
    else:
        is_valid = 1 - is_test

    is_valid = tf.equal(is_valid, 0)

    return image, sketch, class_id, is_valid, category, imagenet_id, sketch_id


def build_input_queue_paired_sketchy(batch_size, data_format='NCHW', distance_map=True, small=True, one_hot=False,
                                     capacity=8192):
    image, sketch, class_id, is_valid, _, _, _ = get_paired_input(
        paired_filenames_1, test_mode=False, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)

    images, sketches, class_ids = tf.train.maybe_shuffle_batch(
        [image, sketch, class_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, min_after_dequeue=32,
        num_threads=4)

    if one_hot:
        labels = map_class_id_to_labels(class_ids)
    else:
        labels = one_hot_to_dense(map_class_id_to_labels(class_ids))
    return images, sketches, labels


def build_input_queue_paired_sketchy_test(batch_size, data_format='NCHW', distance_map=True, small=True, one_hot=False,
                                          capacity=8192):
    image, sketch, class_id, is_valid, category, imagenet_id, sketch_id = get_paired_input(
        paired_filenames_1, test_mode=True, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)

    images, sketches, class_ids, categories, imagenet_ids, sketch_ids = tf.train.maybe_batch(
        [image, sketch, class_id, category, imagenet_id, sketch_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, num_threads=2)

    if one_hot:
        labels = map_class_id_to_labels(class_ids)
    else:
        labels = one_hot_to_dense(map_class_id_to_labels(class_ids))

    return images, sketches, labels, categories, imagenet_ids, sketch_ids


def build_input_queue_paired_flickr(batch_size, data_format='NCHW', distance_map=True, small=True, one_hot=False,
                                    capacity=int(1.5 * 2 ** 15)):
    image, sketch, class_id, is_valid, _, _, _ = get_paired_input(
        paired_filenames_2, test_mode=False, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)

    images, sketches, class_ids = tf.train.maybe_shuffle_batch(
        [image, sketch, class_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, min_after_dequeue=512,
        num_threads=4)

    if one_hot:
        labels = map_class_id_to_labels(class_ids)
    else:
        labels = one_hot_to_dense(map_class_id_to_labels(class_ids))

    return images, sketches, labels


def build_input_queue_paired_mixed(batch_size, proportion=None, data_format='NCHW', distance_map=True, small=True,
                                   one_hot=False, capacity=int(1.5 * 2 ** 15)):
    def _sk_list():
        image_sk, sketch_sk, class_id_sk, is_valid_sk, _, _, _ = get_paired_input(
            paired_filenames_1, test_mode=False, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)
        return image_sk, sketch_sk, class_id_sk, is_valid_sk

    def _f_list():
        image_f, sketch_f, class_id_f, is_valid_f, _, _, _ = get_paired_input(
            paired_filenames_2, test_mode=False, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)
        return image_f, sketch_f, class_id_f, is_valid_f

    idx = tf.floor(tf.random_uniform(shape=(), minval=0., maxval=1., dtype=tf.float32) + proportion)
    sk_list = _sk_list()
    f_list = _f_list()
    image, sketch, class_id, is_valid = [
        tf.cast(a, tf.float32) * idx + tf.cast(b, tf.float32) * (1 - idx) for a, b in zip(sk_list, f_list)
    ]
    class_id = tf.cast(class_id, tf.int64)
    is_valid = tf.cast(is_valid, tf.bool)
    # is_valid = tf.Print(is_valid, [idx, sk_list[4], f_list[4], class_id, sk_list[5], f_list[5], is_valid])

    images, sketches, class_ids = tf.train.maybe_shuffle_batch(
        [image, sketch, class_id],
        batch_size=batch_size, capacity=capacity,
        keep_input=is_valid, min_after_dequeue=512,
        num_threads=4)

    if one_hot:
        labels = map_class_id_to_labels(class_ids)
    else:
        labels = one_hot_to_dense(map_class_id_to_labels(class_ids))

    return images, sketches, labels


def split_inputs(input_data, batch_size, batch_portion, num_gpu):
    input_data_list = []
    dim = len(input_data.get_shape())
    start = 0
    for i in range(num_gpu):
        idx = [start]
        size = [batch_size * batch_portion[i]]
        idx.extend([0] * (dim - 1))
        size.extend([-1] * (dim - 1))
        input_data_list.append(tf.slice(input_data, idx, size))

        start += batch_size * batch_portion[i]
    return input_data_list
