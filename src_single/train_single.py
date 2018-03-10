import os
from time import time
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from graph_single import build_multi_tower_graph, build_single_graph
from input_pipeline import build_input_queue_paired_sketchy, build_input_queue_paired_sketchy_test, build_input_queue_paired_flickr, build_input_queue_paired_mixed
import inception_score
from config import Config

tf.logging.set_verbosity(tf.logging.INFO)
inception_v4_ckpt_path = './inception_v4_model/inception_v4.ckpt'
vgg_16_ckpt_path = './vgg_16_model/vgg_16.ckpt'


def print_parameter_count(verbose=False):
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('generator')
    print(total_parameters)

    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('critic')
    print(total_parameters)


def train(**kwargs):

    def get_inception_score_origin(generator_out, data_format, session, n):
        all_samples = []
        img_dim = 64
        for i in range(n // 100):
            all_samples.append(session.run(generator_out))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, img_dim, img_dim))
        if data_format == 'NCHW':
            all_samples = all_samples.transpose(0, 2, 3, 1)
        return inception_score.get_inception_score(list(all_samples), session)

    status = 0

    # Roll out the parameters
    appendix = Config.resume_from
    batch_size = Config.batch_size
    max_iter_step = Config.max_iter_step
    Diters = Config.disc_iterations
    ld = Config.ld
    optimizer = Config.optimizer
    lr_G = Config.lr_G
    lr_D = Config.lr_D
    num_gpu = Config.num_gpu
    log_dir = Config.log_dir
    ckpt_dir = Config.ckpt_dir
    data_format = Config.data_format
    distance_map = Config.distance_map
    small_img = Config.small_img

    distance_map = distance_map != 0
    small = small_img != 0
    batch_portion = np.array([1, 1, 1, 1], dtype=np.int32)

    iter_from = kwargs['iter_from']

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")

    tf.reset_default_graph()
    print('Iteration starts from: %d' % iter_from)

    assert inception_score.softmax.graph != tf.get_default_graph()
    inception_score._init_inception()

    counter = tf.Variable(initial_value=iter_from, dtype=tf.int32, trainable=False)
    counter_addition_op = tf.assign_add(counter, 1, use_locking=True)

    # proportion = tf.round(tf.cast(counter, tf.float32) / max_iter_step)
    proportion = 0.2 + tf.minimum(0.6, tf.cast(counter, tf.float32) / max_iter_step * 0.6)

    # Construct data queue
    with tf.device('/cpu:0'):
        images, sketches, image_paired_class_ids = build_input_queue_paired_mixed(
            batch_size=batch_size * num_gpu,
            proportion=proportion,
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=2 ** 11)
    with tf.device('/cpu:0'):
        images_d, _, image_paired_class_ids_d = build_input_queue_paired_mixed(
            batch_size=batch_size * num_gpu,
            proportion=tf.constant(0.1, dtype=tf.float32),
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=2 ** 11)
    with tf.device('/cpu:0'):
        _, sketches_100, image_paired_class_ids_100 = build_input_queue_paired_sketchy(
            batch_size=100,
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=1024)

    opt_g, opt_d, loss_g, loss_d, merged_all, gen_out = build_multi_tower_graph(
        images, sketches, images_d,
        sketches_100,
        image_paired_class_ids, image_paired_class_ids_d, image_paired_class_ids_100,
        batch_size=batch_size, num_gpu=num_gpu, batch_portion=batch_portion, training=True,
        learning_rates={
            "generator": lr_G,
            "discriminator": lr_D,
        },
        counter=counter, proportion=proportion, max_iter_step=max_iter_step,
        ld=ld, data_format=data_format,
        distance_map=distance_map,
        optimizer=optimizer)

    inception_score_mean = tf.placeholder(dtype=tf.float32, shape=())
    inception_score_std = tf.placeholder(dtype=tf.float32, shape=())
    inception_score_mean_summary = tf.summary.scalar("inception_score/mean", inception_score_mean)
    inception_score_std_summary = tf.summary.scalar("inception_score/std", inception_score_std)
    inception_score_summary = tf.summary.merge((inception_score_mean_summary, inception_score_std_summary))

    saver = tf.train.Saver()
    try:
        saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV4'))
        perceptual_model_path = inception_v4_ckpt_path
    except:
        try:
            saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16'))
            perceptual_model_path = vgg_16_ckpt_path
        except:
            saver2 = None

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                            intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                            # device_count={"CPU": 8},
                            )
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1   # JIT XLA
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if saver2 is not None:
            saver2.restore(sess, perceptual_model_path)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        if iter_from > 0:
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            summary_writer.reopen()

        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        print_parameter_count(verbose=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run([counter.assign(iter_from)])

        for i in range(iter_from, max_iter_step):
            if status == -1:
                break

            if i % 100 == 0:
                curr_time = time()
                elapsed = curr_time - prev_time
                print(
                    "Now at iteration %d. Elapsed time: %.5fs. Average time: %.5fs/iter" % (i, elapsed, elapsed / 100.))
                prev_time = curr_time

            diters = Diters

            # Train Discriminator
            for j in range(diters):
                # print(j)
                if i % 100 == 0 and j == 0:
                    _, merged, loss_d_out = sess.run([opt_d, merged_all, loss_d],
                                                     options=run_options,
                                                     run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'discriminator_metadata {}'.format(i), i)
                else:
                    _, loss_d_out = sess.run([opt_d, loss_d])
                if np.isnan(np.sum(loss_d_out)):
                    status = -1
                    print("NaN occurred during training D")
                    return status

            # Train Generator
            if i % 100 == 0:
                _, merged, loss_g_out, counter_out, _ = sess.run(
                    [opt_g, merged_all, loss_g, counter, counter_addition_op],
                    options=run_options,
                    run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata {}'.format(i), i)
            else:
                _, loss_g_out, counter_out, _ = sess.run([opt_g, loss_g, counter, counter_addition_op])
            if np.isnan(np.sum(loss_g_out)):
                status = -1
                print("NaN occurred during training G")
                return status

            if i % 5000 == 4999:
                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)

            if i % 1000 == 999:
                this_score = get_inception_score_origin(gen_out, data_format=data_format,
                                                        session=sess, n=10000)
                merged_sum = sess.run(inception_score_summary, feed_dict={
                    inception_score_mean: this_score[0],
                    inception_score_std: this_score[1],
                })
                summary_writer.add_summary(merged_sum, i)

        coord.request_stop()
        coord.join(threads)

    return status


def test(**kwargs):

    def binarize(sketch, threshold=245):
        sketch[sketch < threshold] = 0
        sketch[sketch >= threshold] = 255
        return sketch

    # Roll out the parameters
    appendix = Config.resume_from
    batch_size = Config.batch_size
    log_dir = Config.log_dir
    ckpt_dir = Config.ckpt_dir
    data_format = Config.data_format
    distance_map = Config.distance_map
    small_img = Config.small_img

    build_func = build_single_graph
    channel = 3
    distance_map = distance_map != 0
    small = small_img != 0
    if small:
        img_dim = 64
    else:
        img_dim = 256

    output_folder = os.path.join(log_dir, 'out')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")
    # Construct data queue
    with tf.device('/cpu:0'):
        images, sketches, class_ids, categories, imagenet_ids, sketch_ids = build_input_queue_paired_sketchy_test(
            batch_size=batch_size, data_format=data_format,
            distance_map=distance_map, small=small, capacity=512)

    with tf.device('/gpu:0'):
        ret_list = build_func(images, sketches, None, None,
                              class_ids, None, None,
                              batch_size=batch_size, training=False,
                              data_format=data_format,
                              distance_map=distance_map)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        counter = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while True:
            try:
                generated_img, gt_image, input_sketch, category, imagenet_id, sketch_id = sess.run(
                    [ret_list[0], ret_list[1], ret_list[2], categories, imagenet_ids, sketch_ids])
            except Exception as e:
                print(e.args)
                break

            if counter % 100 == 0:
                curr_time = time()
                elapsed = curr_time - prev_time
                print(
                    "Now at iteration %d. Elapsed time: %.5fs." % (counter, elapsed))
                prev_time = curr_time

            if data_format == 'NCHW':
                generated_img = np.transpose(generated_img, (0, 2, 3, 1))
                gt_image = np.transpose(gt_image, (0, 2, 3, 1))
                input_sketch = np.transpose(input_sketch, (0, 2, 3, 1))
            generated_img = ((generated_img + 1) / 2.) * 255
            gt_image = ((gt_image + 1) / 2.) * 255
            input_sketch = ((input_sketch + 1) / 2.) * 255
            generated_img = generated_img[:, :, :, ::-1].astype(np.uint8)
            gt_image = gt_image[:, :, :, ::-1].astype(np.uint8)
            input_sketch = input_sketch.astype(np.uint8)

            for i in range(batch_size):
                this_prefix = '%s_%d_%d' % (category[i].decode('ascii'),
                                            int(imagenet_id[i].decode('ascii').split('_')[1]),
                                            sketch_id[i])
                img_out_filename = this_prefix + '_fake_B.png'
                img_gt_filename = this_prefix + '_real_B.png'
                sketch_in_filename = this_prefix + '_real_A.png'

                # Save file
                # file_path = os.path.join(output_folder, 'output_%d.jpg' % int(counter / batch_size))
                cv2.imwrite(os.path.join(output_folder, img_out_filename), generated_img[i])
                cv2.imwrite(os.path.join(output_folder, img_gt_filename), gt_image[i])
                cv2.imwrite(os.path.join(output_folder, sketch_in_filename), input_sketch[i])
                # output_img = np.zeros((img_dim * 2, img_dim * batch_size, channel))

                print('Saved file %s' % this_prefix)

                counter += 1

        coord.request_stop()
        coord.join(threads)