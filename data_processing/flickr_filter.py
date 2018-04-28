import imghdr
import itertools
import os
import sys
from time import time
import csv
import PIL.Image as im
import numpy as np
import scipy.io
import scipy.misc as spm

sys.path.append('..')
sys.path.append('../slim')
sys.path.append('../object_detection')
# Notice: you need to clone TF-slim and Tensorflow Object Detection API
# into data_processing:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
# https://github.com/tensorflow/models/tree/master/research/object_detection

import cv2
import coco_data_provider as coco
import tensorflow as tf
from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory

from object_detection.utils import label_map_util


inception_ckpt_path = '../../inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt'

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8), im_width, im_height


def get_imagenet_class_labels():
    synset_list = [s.strip() for s in open('./imagenet_lsvrc_2015_synsets.txt', 'r').readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    synset_to_human_list = open('./imagenet_metadata.txt', 'r').readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


def check_jpg_vadility_single(path):
    if imghdr.what(path) == 'jpg':
        return True
    return False


def check_jpg_vadility(path):
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    invalid_file_list = []
    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")
    for i in range(len(file_list)):
        if i % 5000 == 0:
            curr_time = time()
            elapsed = curr_time - prev_time
            print(
                "Now at iteration %d. Elapsed time: %.5fs." % (i, elapsed))
            prev_time = curr_time
            print(len(invalid_file_list))
        try:
            img = im.open(os.path.join(path, file_list[i]))
            format = img.format.lower()
            if format != 'jpg' and format != 'jpeg':
                raise ValueError
            # img.load()
        except:
            invalid_file_list.append(file_list[i])
    return invalid_file_list


def build_imagenet_graph(path):
    tf.reset_default_graph()
    print(path)

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + "/*.jpg"),
                                                    num_epochs=1, shuffle=False, capacity=100)
    image_reader = tf.WholeFileReader()
    image_file_name, image_file = image_reader.read(filename_queue)

    image = tf.image.decode_jpeg(image_file, channels=3, fancy_upscaling=True)

    model_name = 'inception_resnet_v2'
    network_fn = nets_factory.get_network_fn(model_name, is_training=False, num_classes=1001)

    preprocessing_name = model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

    eval_image_size = network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    filenames, images = tf.train.batch([image_file_name, image], batch_size=100, num_threads=2, capacity=500)
    logits, _ = network_fn(images)

    variables_to_restore = slim.get_variables_to_restore()
    predictions = tf.argmax(logits, 1)

    return filenames, logits, predictions, variables_to_restore


def filter_by_imagenet(path, cls_name):
    labels_dict = get_imagenet_class_labels()
    filenames, logits, predictions, variables_to_restore = build_imagenet_graph(path)
    saver = tf.train.Saver(variables_to_restore)
    output_filename_list = []
    counter = 0

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, inception_ckpt_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while True:
            try:
                filename_list, logit_array, prediction_list = sess.run([filenames, logits, predictions])
            except Exception as e:
                break

            if counter % 5000 == 0:
                print("Evaluated %d files" % counter)
                print(len(output_filename_list))

            prediction_dict = {os.path.split(filename)[1]: labels_dict[prediction] for filename, prediction in
                               zip(filename_list, prediction_list)}
            for i, j in prediction_dict.items():
                j = [p.strip() for p in j.lower().split(',')]
                if cls_name.lower() in j and len(j) == 1:
                    output_filename_list.append(i.decode('ascii'))

            counter += 100

        coord.request_stop()
        coord.join(threads)

        return output_filename_list


# SSD filter for COCO classes in Tensorflow instead of Caffe.
# Not fully functional yet. It will not output filtered filenames.
def filter_by_coco(path, cls_name):
    TEST_IMAGE_PATHS = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    counter = 0
    output_filename_list = []

    tf.reset_default_graph()
    print(path)

    PATH_TO_CKPT = '../ssd_inception_v2/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('../../object_detection/data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    # Label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    for i in range(80):
        if categories[i]['name'] == cls_name:
            cls_index = categories[i]['id']

    # Load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # Input queue
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + "/*.jpg"),
                                                        num_epochs=1, shuffle=False, capacity=100)
        image_reader = tf.WholeFileReader()
        image_file_name, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file, channels=3, fancy_upscaling=True)
        image0 = tf.image.resize_image_with_crop_or_pad(image, 500, 500)
        image = tf.image.resize_images(image0, [250, 250], method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.uint8)

        filenames, images = tf.train.batch([image_file_name, image], batch_size=20, num_threads=2, capacity=500)

        # Graph Def
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='', input_map={'image_tensor:0': images})

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                if counter % 5 == 0:
                    curr_time = time()
                    elapsed = curr_time - prev_time
                    print(
                        "Now at iteration %d. Elapsed time: %.5fs." % (counter, elapsed))
                    prev_time = curr_time

                image = im.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np, im_width, im_height = load_image_into_numpy_array(image)
                image_np = scipy.misc.imresize(image_np, 0.5, 'bilinear')
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Filter results
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.int32),
                scores = np.squeeze(scores)
                idx = np.logical_and(scores > 0.9, classes == cls_index)
                portion = np.prod(boxes[idx], axis=1) / (im_width * im_height)

                if portion.size > 0:
                    print()

                counter += 1


def filter_images(flickr_dir, cls_name):
    imagenet_classes = [i[:-1] for i in open('./imagenet_share_classes.txt').readlines()]
    coco_classes = [i[:-1] for i in open('./coco_share_classes.txt').readlines()]

    this_dir = os.path.join(flickr_dir, cls_name)

    invalid_file_list = []
    print("Invalid file number: %d" % len(invalid_file_list))
    for file_name in invalid_file_list:
        os.remove(os.path.join(this_dir, file_name))

    if cls_name in imagenet_classes:
        output_filename_list = filter_by_imagenet(this_dir, cls_name)
    elif cls_name in coco_classes:
        output_filename_list = filter_by_coco(this_dir, cls_name)
    else:
        raise NotImplementedError

    file_list = [f for f in os.listdir(this_dir) if os.path.isfile(os.path.join(this_dir, f))]
    print(len(file_list) - len(output_filename_list))
    for file_name in file_list:
        if file_name not in output_filename_list:
            os.remove(os.path.join(this_dir, file_name))


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                        intra_op_parallelism_threads=4)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with open('./imagenet_share_classes.txt', 'r') as f:
    classes_list = [i[:-1] for i in f.readlines()]

if __name__ == '__main__':
    # # Imagenet
    # labels_dict = get_imagenet_class_labels()
    # labels_list = [label + '\n' for i, label in labels_dict.items()]
    # with open('./imagenet_classes.txt', 'w') as f:
    #     f.writelines(labels_list)

    # COCO
    labels_list = [cls['name'] for cls in coco.get_all_images_data_categories(split='train')[2]]

    # with open('./all_classes', 'r') as f:
    #     sketchy_class_list = [i[:-1] for i in f.readlines()]
    #
    # filtered_classes = []
    # for cls in sketchy_class_list:
    #     for large_cls in labels_list:
    #         large_cls_names = [i.strip() for i in large_cls.split(',')]
    #         for name in large_cls_names:
    #             if cls.lower() == name.lower() and cls.lower() not in filtered_classes:
    #                 filtered_classes.append(cls + '\n')
    #
    # with open('./coco_share_classes.txt', 'w') as f:
    #     f.writelines(filtered_classes)

    # Inference
    filter_range = (8, 12)

    class_list = ['airplane']
    for class_name in class_list:
        filter_images('../flickr_output', class_name)
