import multiprocessing as mp
import os
import sys
import csv
import numpy as np
# import scipy.io
# import scipy.misc as spm

import cv2
from scipy import ndimage
import tensorflow as tf
# from tensorflow.python.framework import ops


np.seterr(all='raise')


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


valid_class_names = ['car_(sedan)']    # Class to convert

classes_info = '../data_processing/classes.csv'
photo_folder = '../flickr_coco'
sketch_folder = '../flickr_hed/jpg'
data_dir = '../flickr_output'

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                        intra_op_parallelism_threads=8)


def check_repeat(seq):
    seen = set()
    seen_add = seen.add
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    return list(seen_twice)


def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        l = list(reader)

    return l


def read_txt(filename):
    with open(filename) as txtfile:
        lines = txtfile.readlines()
    return [l[:-1] for l in lines]


def build_graph():
    photo_filename = tf.placeholder(dtype=tf.string, shape=())
    label_filename = tf.placeholder(dtype=tf.string, shape=())
    photo = tf.read_file(photo_filename)
    label = tf.read_file(label_filename)
    photo_decoded = tf.image.decode_jpeg(photo, channels=3, fancy_upscaling=True)
    label_decoded = tf.image.decode_png(label)

    # Encode 64x64
    photo_input = tf.placeholder(dtype=tf.uint8, shape=(256, 256, 3))
    photo_small_input = tf.placeholder(dtype=tf.uint8, shape=(64, 64, 3))
    label_input = tf.placeholder(dtype=tf.uint8, shape=(256, 256, 1))
    label_small_input = tf.placeholder(dtype=tf.uint8, shape=(64, 64, 1))

    photo_stream = tf.image.encode_jpeg(photo_input, quality=95, progressive=False,
                                        optimize_size=False, chroma_downsampling=False)
    photo_small_stream = tf.image.encode_jpeg(photo_small_input, quality=95, progressive=False,
                                              optimize_size=False, chroma_downsampling=False)
    label_stream = tf.image.encode_png(label_input, compression=7)
    label_small_stream = tf.image.encode_png(label_small_input, compression=7)

    return photo_filename, label_filename, photo, label, photo_decoded, label_decoded,\
        photo_input, photo_small_input, label_input, label_small_input, photo_stream, photo_small_stream,\
        label_stream, label_small_stream


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
    dir_list = [d for d in os.listdir(sketch_folder) if os.path.isdir(os.path.join(sketch_folder, d))]
    classes = read_csv(classes_info)
    classes_ids = [item['Name'] for item in classes]
    work_list = []

    for dir in dir_list:
        if dir not in valid_class_names:
            continue
        this_sketch_folder = os.path.join(sketch_folder, dir)
        this_photo_folder = os.path.join(photo_folder, dir)
        sketch_files_list = [f for f in os.listdir(this_sketch_folder)
                             if os.path.isfile(os.path.join(this_sketch_folder, f))]
        photo_files_list = [f for f in os.listdir(this_photo_folder)
                            if os.path.isfile(os.path.join(this_photo_folder, f)) and
                            os.path.isfile(os.path.join(this_sketch_folder, os.path.splitext(f)[0] + '.png'))]
        assert len(photo_files_list) == len(sketch_files_list)

        class_id = classes_ids.index(dir)
        work_list.append((class_id, photo_files_list, sketch_files_list))

    num_processes = 8

    # launch processes
    pool = mp.Pool(processes=num_processes)
    results = []
    for i in range(len(work_list)):
        result = pool.apply_async(write_dir_photo, args=(work_list[i], classes_ids, i))
        results.append(result)
    for i in range(len(results)):
        results[i].get()

    pool.close()
    pool.join()


def write_dir_photo(object_item, classes_ids, process_id):

    class_id_num = object_item[0]
    class_id = str(class_id_num)
    Category = classes_ids[class_id_num]
    if Category not in valid_class_names:
        return

    photo_files = object_item[1]
    sketch_files = object_item[2]

    processed_num = -1
    writer = None
    file_contain_photo_num = 2048

    path_image = os.path.join(photo_folder, Category)
    path_label = os.path.join(sketch_folder, Category)

    with tf.device('/cpu:0'):
        photo_filename, label_filename, photo, label, photo_decoded, label_decoded, \
            photo_input, photo_small_input, label_input, label_small_input, photo_stream, photo_small_stream, \
            label_stream, label_small_stream = build_graph()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('ID %s with num photos: %d' % (class_id, len(photo_files)))

        for i in range(len(photo_files)):

            processed_num += 1

            if processed_num % file_contain_photo_num == 0:
                if writer is not None:
                    writer.close()
                    print('ID %s current at: %d' % (class_id, i))
                else:
                    print('Init first writer')
                writer = tf.python_io.TFRecordWriter(
                    os.path.join(data_dir, Category + '_coco_seg_%d.tfrecord' % (processed_num // file_contain_photo_num)))

            cur_photo_path = os.path.join(path_image, photo_files[i])
            label_name = os.path.splitext(photo_files[i])[0] + '.png'
            if label_name not in sketch_files:
                print('Wrong filename: %s' % photo_files[i])
            cur_label_path = os.path.join(path_label, label_name)

            try:
                out_image, out_image_decoded = sess.run([photo, photo_decoded], feed_dict={
                    photo_filename: os.path.join(cur_photo_path)})
                out_label, out_label_decoded = sess.run([label, label_decoded], feed_dict={
                    label_filename: os.path.join(cur_label_path)})
            except:
                print('Invalid file')
                continue

            # Resize
            channel_num = 3. if len(out_label_decoded.shape) == 3 and out_label_decoded.shape[2] == 3 else 1.
            out_image_decoded = cv2.resize(out_image_decoded, (256, 256), interpolation=cv2.INTER_AREA)
            out_image_decoded_small = cv2.resize(out_image_decoded, (64, 64), interpolation=cv2.INTER_AREA)
            out_label_decoded = (np.sum(out_label_decoded.astype(np.float64), axis=2)/channel_num).astype(np.uint8)
            out_label_decoded_small = cv2.resize(out_label_decoded, (64, 64), interpolation=cv2.INTER_AREA)
            if (out_label_decoded_small == 0).all() or (out_label_decoded_small == 255).all():
                print('Warning: blank sketch from resize')
                continue

            # Distance map
            out_dist_map = ndimage.distance_transform_edt(binarize(out_label_decoded))
            out_dist_map = (out_dist_map / out_dist_map.max() * 255.).astype(np.uint8)

            out_dist_map_small = ndimage.distance_transform_edt(binarize(out_label_decoded_small))
            out_dist_map_small = (out_dist_map_small / out_dist_map_small.max() * 255.).astype(np.uint8)

            # Stream
            image_string, label_string = sess.run([photo_stream, label_stream], feed_dict={
                photo_input: out_image_decoded, label_input: out_label_decoded.reshape((256, 256, 1))
            })
            image_string_small, label_string_small = sess.run([photo_small_stream, label_small_stream], feed_dict={
                photo_small_input: out_image_decoded_small, label_small_input: out_label_decoded_small.reshape((64, 64, 1))
            })
            dist_map_string = sess.run(label_stream, feed_dict={label_input: out_dist_map.reshape((256, 256, 1))})
            dist_map_string_small = sess.run(label_small_stream, feed_dict={
                label_small_input: out_dist_map_small.reshape((64, 64, 1))})

            example = tf.train.Example(features=tf.train.Features(feature={
                'ImageNetID': _bytes_feature(''.encode('utf-8')),
                'SketchID': _int64_feature(0),
                'Category': _bytes_feature(Category.encode('utf-8')),
                'CategoryID': _int64_feature(class_id_num),
                'Difficulty': _int64_feature(0),
                'Stroke_Count': _int64_feature(0),
                'WrongPose': _int64_feature(0),
                'Context': _int64_feature(0),
                'Ambiguous': _int64_feature(0),
                'Error': _int64_feature(0),
                'is_test': _int64_feature(0),
                'class_id': _int64_feature(class_id_num),
                'image_jpeg': _bytes_feature(image_string),
                'image_small_jpeg': _bytes_feature(image_string_small),
                'sketch_png': _bytes_feature(label_string),
                'sketch_small_png': _bytes_feature(label_string_small),
                'dist_map_png': _bytes_feature(dist_map_string),
                'dist_map_small_png': _bytes_feature(dist_map_string_small),
            }))
            writer.write(example.SerializeToString())

        writer.close()


write_image_data()
