import os
import sys
import csv
import numpy as np
import scipy.io
import scipy.misc as spm

import cv2
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops


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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


classes_info = '../data_processing/classes.csv'
photo_folder = '../Datasets/Sketchy/rendered_256x256/256x256/photo/tx_000000000000'
sketch_folder = '../Datasets/Sketchy/rendered_256x256/256x256/sketch/tx_000000000000'
info_dir = '../Datasets/Sketchy/info'
data_dir = '../tfrecords/sketchy'

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                        intra_op_parallelism_threads=4)


def check_repeat(seq):
    seen = set()
    seen_add = seen.add
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    return list(seen_twice)


def build_graph():
    photo_filename = tf.placeholder(dtype=tf.string, shape=())
    label_filename = tf.placeholder(dtype=tf.string, shape=())
    photo = tf.read_file(photo_filename)
    label = tf.read_file(label_filename)
    photo_decoded = tf.image.decode_jpeg(photo, fancy_upscaling=True)
    label_decoded = tf.image.decode_png(label)

    # Encode 64x64
    photo_input = tf.placeholder(dtype=tf.uint8, shape=(64, 64, 3))
    label_input = tf.placeholder(dtype=tf.uint8, shape=(256, 256, 1))
    label_small_input = tf.placeholder(dtype=tf.uint8, shape=(64, 64, 1))

    photo_stream = tf.image.encode_jpeg(photo_input, quality=95, progressive=False,
                                        optimize_size=False, chroma_downsampling=False)
    label_stream = tf.image.encode_png(label_input, compression=7)
    label_small_stream = tf.image.encode_png(label_small_input, compression=7)

    return photo_filename, label_filename, photo, label, photo_decoded, label_decoded, photo_input, label_input,\
           label_small_input, photo_stream, label_stream, label_small_stream


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


def binarize(sketch, threshold=245):
    sketch[sketch < threshold] = 0
    sketch[sketch >= threshold] = 255
    return sketch


def write_image_data():

    csv_file = os.path.join(info_dir, 'stats.csv')
    stat_info = read_csv(csv_file)
    classes = read_csv(classes_info)
    classes_ids = [item['Name'] for item in classes]

    test_list = read_txt(os.path.join(info_dir, 'testset.txt'))

    invalid_notations = ['invalid-ambiguous.txt', 'invalid-context.txt', 'invalid-error.txt', 'invalid-pose.txt']
    invalid_files = []
    for txtfile in invalid_notations:
        cur_path = os.path.join(info_dir, txtfile)
        files = read_txt(cur_path)
        files = [f[:-1] for f in files]
        invalid_files.extend(files)

    path_image = photo_folder
    path_label = sketch_folder

    dirs, stats = split_csvlist(stat_info)
    photo_filename, label_filename, photo, label, photo_decoded, label_decoded, photo_input, label_input, \
    label_small_input, photo_stream, label_stream, label_small_stream = build_graph()
    assert len(dirs) == len(stats)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(len(dirs)):
            dir = dirs[i].replace(' ', '_')
            print(dir)
            class_id = classes_ids.index(dir)
            stat = stats[i]
            writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, dir + '.tfrecord'))

            cur_photo_path = os.path.join(path_image, dir)
            cur_label_path = os.path.join(path_label, dir)
            num_label = len(stat)
            # photo_files = [f for f in os.listdir(cur_photo_path) if os.path.isfile(os.path.join(cur_photo_path, f))]
            # label_files = [f for f in os.listdir(cur_label_path) if os.path.isfile(os.path.join(cur_label_path, f))]

            for j in range(num_label):
                if j % 500 == 499:
                    print(j)
                item = stat[j]

                ImageNetID = item['ImageNetID']
                SketchID = int(item['SketchID'])
                Category = item['Category']
                CategoryID = int(item['CategoryID'])
                Difficulty = int(item['Difficulty'])
                Stroke_Count = int(item['Stroke_Count'])

                WrongPose = int(item['WrongPose?'])
                Context = int(item['Context?'])
                Ambiguous = int(item['Ambiguous?'])
                Error = int(item['Error?'])

                if os.path.join(dir, ImageNetID + '.jpg') in test_list:
                    IsTest = 1
                else:
                    IsTest = 0

                # print(os.path.join(cur_photo_path, ImageNetID + '.jpg'))
                # print(os.path.join(cur_label_path, ImageNetID + '-' + str(SketchID) + '.png'))
                out_image, out_image_decoded = sess.run([photo, photo_decoded], feed_dict={
                    photo_filename: os.path.join(cur_photo_path, ImageNetID + '.jpg')})
                out_label, out_label_decoded = sess.run([label, label_decoded], feed_dict={
                    label_filename: os.path.join(cur_label_path, ImageNetID + '-' + str(SketchID) + '.png')})

                # Resize
                out_image_decoded_small = cv2.resize(out_image_decoded, (64, 64), interpolation=cv2.INTER_AREA)
                out_label_decoded = (np.sum(out_label_decoded.astype(np.float64), axis=2)/3).astype(np.uint8)
                out_label_decoded_small = cv2.resize(out_label_decoded, (64, 64), interpolation=cv2.INTER_AREA)

                # Distance map
                out_dist_map = ndimage.distance_transform_edt(binarize(out_label_decoded))
                out_dist_map = (out_dist_map / out_dist_map.max() * 255.).astype(np.uint8)

                out_dist_map_small = ndimage.distance_transform_edt(binarize(out_label_decoded_small))
                out_dist_map_small = (out_dist_map_small / out_dist_map_small.max() * 255.).astype(np.uint8)

                # Stream
                image_string_small, label_string_small = sess.run([photo_stream, label_small_stream], feed_dict={
                    photo_input: out_image_decoded_small, label_small_input: out_label_decoded_small.reshape((64, 64, 1))
                })
                dist_map_string = sess.run(label_stream, feed_dict={label_input: out_dist_map.reshape((256, 256, 1))})
                dist_map_string_small = sess.run(label_small_stream, feed_dict={
                    label_small_input: out_dist_map_small.reshape((64, 64, 1))})

                example = tf.train.Example(features=tf.train.Features(feature={
                    'ImageNetID': _bytes_feature(ImageNetID.encode('utf-8')),
                    'SketchID': _int64_feature(SketchID),
                    'Category': _bytes_feature(Category.encode('utf-8')),
                    'CategoryID': _int64_feature(CategoryID),
                    'Difficulty': _int64_feature(Difficulty),
                    'Stroke_Count': _int64_feature(Stroke_Count),
                    'WrongPose': _int64_feature(WrongPose),
                    'Context': _int64_feature(Context),
                    'Ambiguous': _int64_feature(Ambiguous),
                    'Error': _int64_feature(Error),
                    'is_test': _int64_feature(IsTest),
                    'class_id': _int64_feature(class_id),
                    'image_jpeg': _bytes_feature(out_image),
                    'image_small_jpeg': _bytes_feature(image_string_small),
                    'sketch_png': _bytes_feature(out_label),
                    'sketch_small_png': _bytes_feature(label_string_small),
                    'dist_map_png': _bytes_feature(dist_map_string),
                    'dist_map_small_png': _bytes_feature(dist_map_string_small),
                }))
                writer.write(example.SerializeToString())

                # coord.request_stop()
                # coord.join(threads)

            writer.close()


write_image_data()
