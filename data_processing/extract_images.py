import os
import cv2
import numpy as np
import tensorflow as tf

datafile_path = "../flickr_output"
image_output_path = "../extract_output/images"
edgemap_output_path = "../extract_output/edges"


def get_paired_input(filenames):
    filename_queue = tf.train.string_input_producer(filenames, capacity=512, shuffle=False, num_epochs=1)
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
            'sketch_png': tf.FixedLenFeature([], tf.string),
        }
    )

    image = features['image_jpeg']
    sketch = features['sketch_png']

    # Attributes
    category = features['Category']
    # Not used
    # imagenet_id = features['ImageNetID']
    # sketch_id = features['SketchID']
    # class_id = features['class_id']
    # is_test = features['is_test']
    # Stroke_Count = features['Stroke_Count']
    # Difficulty = features['Difficulty']
    # CategoryID = features['CategoryID']
    # WrongPose = features['WrongPose']
    # Context = features['Context']
    # Ambiguous = features['Ambiguous']
    # Error = features['Error']

    return image, sketch, category


def build_queue(filenames, batch_size, capacity=1024):
    image, sketch, category = get_paired_input(filenames)

    images, sketchs, categories = tf.train.batch(
        [image, sketch, category],
        batch_size=1, capacity=capacity, num_threads=2, allow_smaller_final_batch=True)

    return images, sketchs, categories


def extract_images(class_name):
    filenames = sorted([os.path.join(datafile_path, f) for f in os.listdir(datafile_path)
                        if os.path.isfile(os.path.join(datafile_path, f)) and f.startswith(class_name)])

    # Make dirs
    this_image_path = os.path.join(image_output_path, class_name)
    this_edgemap_path = os.path.join(edgemap_output_path, class_name)
    if not os.path.isdir(image_output_path) and not os.path.exists(image_output_path):
        os.makedirs(image_output_path)
    if not os.path.isdir(this_image_path) and not os.path.exists(this_image_path):
        os.makedirs(this_image_path)
    if not os.path.isdir(edgemap_output_path) and not os.path.exists(edgemap_output_path):
        os.makedirs(edgemap_output_path)
    if not os.path.isdir(this_edgemap_path) and not os.path.exists(this_edgemap_path):
        os.makedirs(this_edgemap_path)

    # Read tfrecords
    images, sketchs, categories = build_queue(filenames, 64)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        counter = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while True:
            try:
                raw_jpeg_data, raw_png_data, category_names = sess.run(
                    [images, sketchs, categories])
                filename_appendix = "_%08d" % counter
                with open(os.path.join(this_image_path, class_name + filename_appendix + '.jpg'), 'wb') as f:
                    f.write(raw_jpeg_data[0])
                with open(os.path.join(this_edgemap_path, class_name + filename_appendix + '.png'), 'wb') as f:
                    f.write(raw_png_data[0])

                counter += 1
            except Exception as e:
                print(e.args)
                break

            if counter % 100 == 0:
                print("Now at iteration %d." % counter)

        coord.request_stop()
        coord.join(threads)
    print()


if __name__ == "__main__":
    # class_name = "airplane"
    filenames = sorted([f for f in os.listdir(datafile_path) if os.path.isfile(os.path.join(datafile_path, f))])
    class_names = sorted(list({f.replace('_', '.').split('.', 1)[0] for f in filenames}))
    print('Num of classes found: %d' % len(class_names))

    for cls in class_names:
        extract_images(cls)
