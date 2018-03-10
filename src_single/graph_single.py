import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

import models_mru as models
from input_pipeline import *
from inception_v4 import inception_v4_base, inception_v4, inception_v4_arg_scope
# from vgg import vgg_16, vgg_19, vgg_arg_scope
from sn import spectral_normed_weight
from config import Config

slim = tf.contrib.slim


def dist_map_to_image(input, threshold=0.015):
    ret = tf.cast(1 - tf.cast(tf.less(input + 1, threshold), tf.int32), tf.float32)
    return ret


def compute_gradients(losses, optimizers, var_lists):
    assert len(losses) == len(optimizers) and len(optimizers) == len(var_lists)
    grads = []
    for i in range(len(losses)):
        this_grad = optimizers[i].compute_gradients(losses[i], var_list=var_lists[i])
        grads.append(this_grad)
    return grads


def average_gradients(tower_grads_list):
    """notice: Variable pointers come from the first tower"""

    grads_list = []
    for i in range(len(tower_grads_list)):
        average_grads = []
        tower_grads = tower_grads_list[i]
        num_towers = len(tower_grads)
        for grad_and_vars in zip(*tower_grads):
            grads = []
            grad = 'No Value'
            if grad_and_vars[0][0] is None:
                all_none = True
                for j in range(num_towers):
                    if grad_and_vars[j][0] is not None:
                        all_none = False
                if all_none:
                    grad = None
                else:
                    raise ValueError("None gradient inconsistent between towers.")
            else:
                for g, _ in grad_and_vars:
                    expanded_grad = tf.expand_dims(g, axis=0)
                    grads.append(expanded_grad)

                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, axis=0)

            v = grad_and_vars[0][1]
            if isinstance(grad, str):
                raise ValueError("Gradient not defined when averaging.")
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        grads_list.append(average_grads)
    return grads_list


def gather_summaries(max_outputs=100):
    # Image summaries
    orig_img_sum = tf.summary.image("original_img", tf.get_collection("original_img")[0], max_outputs=max_outputs)
    orig_img_sum_d = tf.summary.image("original_img_d", tf.get_collection("original_img_d")[0], max_outputs=max_outputs)
    orig_img_sum2 = tf.summary.image("original_img_2", tf.get_collection("original_img_2")[0], max_outputs=max_outputs)
    img_sum_2t1 = tf.summary.image("img_2_to_1", tf.get_collection("img_2_to_1")[0], max_outputs=max_outputs)
    img_sum_2t1_b = tf.summary.image("img_2_to_1_b", tf.get_collection("img_2_to_1_b")[0], max_outputs=max_outputs)
    if len(tf.get_collection("dist_map_img_2")) > 0:
        dist_map_sum_2 = tf.summary.image("dist_map_img_2", tf.get_collection("dist_map_img_2")[0], max_outputs=max_outputs)

    # Scalar
    tf.summary.scalar("GAN_loss/G", tf.reduce_mean(tf.get_collection("GAN_loss_g")))
    tf.summary.scalar("GAN_loss/D", tf.reduce_mean(tf.get_collection("GAN_loss_d")))
    tf.summary.scalar("GAN_loss/GP", tf.reduce_mean(tf.get_collection("GAN_loss_d_gp")))

    tf.summary.scalar("ACGAN_loss/G", tf.reduce_mean(tf.get_collection("ACGAN_loss_g")))
    tf.summary.scalar("ACGAN_loss/D", tf.reduce_mean(tf.get_collection("ACGAN_loss_d")))

    tf.summary.scalar("direct_loss", tf.reduce_mean(tf.get_collection("direct_loss")))
    tf.summary.scalar("diversity_loss", tf.reduce_mean(tf.get_collection("diversity_loss")))
    tf.summary.scalar("DECAY_loss/G", tf.reduce_mean(tf.get_collection("weight_decay_loss_g")))
    tf.summary.scalar("DECAY_loss/D", tf.reduce_mean(tf.get_collection("weight_decay_loss_d")))
    tf.summary.scalar("sketch_proportion", tf.reduce_mean(tf.get_collection("sketch_proportion")))

    tf.summary.scalar("total_loss/g", tf.reduce_mean(tf.get_collection("total_loss_g")))
    tf.summary.scalar("total_loss/d", tf.reduce_mean(tf.get_collection("total_loss_d")))

    return tf.summary.merge_all()


def gather_losses():
    loss_g = tf.reduce_mean(tf.get_collection("loss_g"))
    loss_d = tf.reduce_mean(tf.get_collection("loss_d"))
    return loss_g, loss_d


def build_multi_tower_graph(images, sketches, images_d,
                            sketches_100,
                            image_paired_class_ids, image_paired_class_ids_d, image_paired_class_ids_100,
                            batch_size, num_gpu, batch_portion, training,
                            learning_rates, counter, proportion,
                            max_iter_step,
                            ld=10,
                            data_format='NCHW', distance_map=True,
                            optimizer='Adam'):
    models.set_param(data_format=data_format)
    tf.add_to_collection("sketch_proportion", proportion)

    with tf.device('/cpu:0'):
        images_list = split_inputs(images, batch_size, batch_portion, num_gpu)
        images_d_list = split_inputs(images_d, batch_size, batch_portion, num_gpu)
        sketches_list = split_inputs(sketches, batch_size, batch_portion, num_gpu)
        image_paired_class_ids_list = split_inputs(image_paired_class_ids, batch_size, batch_portion, num_gpu)
        image_paired_class_ids_d_list = split_inputs(image_paired_class_ids_d, batch_size, batch_portion, num_gpu)
        sketches_100_list = [tf.identity(sketches_100)] * len(batch_portion)
        image_paired_class_ids_100_list = [tf.identity(image_paired_class_ids_100)] * len(batch_portion)

    lr_g = learning_rates['generator']
    lr_d = learning_rates['discriminator']
    optimizer = get_optimizer(optimizer)
    decay = tf.maximum(0.2, 1. - (tf.cast(counter, tf.float32) / max_iter_step * 0.9))
    optim_g = optimizer(learning_rate=lr_g * decay)
    optim_d = optimizer(learning_rate=lr_d * decay)

    tower_grads_g = []
    tower_grads_d = []
    for i in range(num_gpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('GPU', i)) as scope:
                print('GPU %d' % i)
                loss_g, loss_d, grad_g, grad_d, inception_gen_out = build_single_graph(images_list[i],
                                                                                       sketches_list[i],
                                                                                       images_d_list[i],
                                                                                       sketches_100_list[i],
                                                                                       image_paired_class_ids_list[i],
                                                                                       image_paired_class_ids_d_list[i],
                                                                                       image_paired_class_ids_100_list[
                                                                                           i],
                                                                                       batch_size * batch_portion[i],
                                                                                       training,
                                                                                       ld=ld, data_format=data_format,
                                                                                       distance_map=distance_map,
                                                                                       optim_g=optim_g,
                                                                                       optim_d=optim_d)

                tower_grads_g.append(grad_g)
                tower_grads_d.append(grad_d)

    assert len(tower_grads_g) == len(tower_grads_d)
    if len(tower_grads_d) == 1:
        ave_grad_g = grad_g
        ave_grad_d = grad_d
    else:
        ave_grad_g, ave_grad_d = average_gradients((tower_grads_g, tower_grads_d))

    # Apply gradients
    tf.get_variable_scope()._reuse = False    # Hack to force initialization of optimizer variables

    if Config.sn:
        # Get the update ops
        spectral_norm_update_ops = tf.get_collection(Config.SPECTRAL_NORM_UPDATE_OPS)
    else:
        spectral_norm_update_ops = [tf.no_op()]
        assign_ops = tf.no_op()

    # Clip gradients if using WGAN/DRAGAN
    global_grad_norm_G = None
    global_grad_norm_G_clipped = None
    global_grad_norm_D = None
    global_grad_norm_D_clipped = None

    if not Config.sn:
        max_grad_norm_G = 50.
        max_grad_norm_D = 100.
        hard_clip_norm_G = 5.
        hard_clip_norm_D = 10.

        ave_grad_g_tensors, ave_grad_g_vars = list(zip(*ave_grad_g))
        global_grad_norm_G = clip_ops.global_norm(ave_grad_g_tensors)
        ave_grad_g_tensors, _ = clip_ops.clip_by_global_norm(ave_grad_g_tensors, max_grad_norm_G, global_grad_norm_G)
        ave_grad_g_tensors = [clip_ops.clip_by_norm(t, hard_clip_norm_G) for t in ave_grad_g_tensors]
        ave_grad_g = list(zip(ave_grad_g_tensors, ave_grad_g_vars))

        ave_grad_d_tensors, ave_grad_d_vars = list(zip(*ave_grad_d))
        global_grad_norm_D = clip_ops.global_norm(ave_grad_d_tensors)
        ave_grad_d_tensors, _ = clip_ops.clip_by_global_norm(ave_grad_d_tensors, max_grad_norm_D, global_grad_norm_D)
        ave_grad_d_tensors = [clip_ops.clip_by_norm(t, hard_clip_norm_D) for t in ave_grad_d_tensors]
        ave_grad_d = list(zip(ave_grad_d_tensors, ave_grad_d_vars))
    with tf.control_dependencies(spectral_norm_update_ops):
        opt_g = optimize(ave_grad_g, optim_g, None, 'gradient_norm', global_norm=global_grad_norm_G,
                         global_norm_clipped=global_grad_norm_G_clipped, appendix='_G')
    opt_d = optimize(ave_grad_d, optim_d, None, 'gradient_norm', global_norm=global_grad_norm_D,
                     global_norm_clipped=global_grad_norm_D_clipped, appendix='_D')

    summaries = gather_summaries()
    loss_g, loss_d = gather_losses()

    # Generator output from last tower
    return opt_g, opt_d, loss_g, loss_d, summaries, inception_gen_out


def build_single_graph(images, sketches, images_d,
                       sketches_100,
                       image_data_class_id, image_data_class_id_d, image_data_2_class_id_100,
                       batch_size, training,
                       ld=10,
                       data_format='NCHW', distance_map=True,
                       optim_g=None, optim_d=None):

    def transfer(image_data, labels, num_classes, reuse=False, data_format=data_format, output_channel=3):

        generator_scope = 'generator'

        image_gen, noise = generator(image_data, output_channel=output_channel, num_classes=num_classes,
                                     reuse=reuse, data_format=data_format, labels=labels,
                                     scope_name=generator_scope)

        return image_gen, noise, labels

    models.set_param(data_format=data_format)
    num_classes = get_num_classes()

    ############################# Graph #################################
    # Input
    generator = models.generator
    discriminator = models.critic

    image_gens, image_gens_noise, image_labels = transfer(sketches, image_data_class_id,
                                                          num_classes=num_classes, reuse=False,
                                                          data_format=data_format,
                                                          output_channel=3)
    image_gens_b, image_gens_noise_b, image_labels_b = transfer(sketches, image_data_class_id,
                                                                num_classes=num_classes, reuse=True,
                                                                data_format=data_format,
                                                                output_channel=3)

    if not training:
        return image_gens, images, sketches

    # Inception Generation
    image_gen_100, _, _ = transfer(sketches_100, image_data_2_class_id_100, num_classes=num_classes,
                                   reuse=True, data_format=data_format, output_channel=3)

    # Discriminator
    real_disc_out, real_logit = discriminator(images_d, num_classes=num_classes, labels=image_data_class_id_d,
                                              reuse=False, data_format=data_format, scope_name='discriminator')
    fake_disc_out, fake_logit = discriminator(image_gens, num_classes=num_classes, labels=image_labels,
                                              reuse=True, data_format=data_format, scope_name='discriminator')
    ############################# End Graph ##############################

    loss_g, loss_d = get_losses(discriminator, None,
                                num_classes, data_format, ld,
                                # images
                                images, sketches,
                                images_d,
                                image_gens, image_gens_b,
                                # latent and labels
                                image_data_class_id, image_data_class_id_d,
                                image_gens_noise, image_gens_noise_b,
                                image_labels, image_labels_b,
                                # critic out
                                real_disc_out, fake_disc_out,
                                # logit
                                real_logit, fake_logit,
                                )

    if data_format == 'NCHW':
        tf.add_to_collection("original_img", tf.transpose(images, (0, 2, 3, 1)))
        tf.add_to_collection("original_img_d", tf.transpose(images_d, (0, 2, 3, 1)))
        tf.add_to_collection("original_img_2", tf.transpose(sketches, (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1", tf.transpose(image_gens, (0, 2, 3, 1)))
        tf.add_to_collection("img_2_to_1_b", tf.transpose(image_gens_b, (0, 2, 3, 1)))
        if distance_map:
            tf.add_to_collection("dist_map_img_2",
                                 tf.transpose(dist_map_to_image(sketches, threshold=0.025), (0, 2, 3, 1)))

    # Add loss to collections
    tf.add_to_collection("loss_g", loss_g)
    tf.add_to_collection("loss_d", loss_d)

    # Variable Collections
    var_collections = {
        'generator': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
        'discriminator': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
    }

    ############# Reuse Variables for next tower (?) #############
    tf.get_variable_scope().reuse_variables()
    ############# Reuse Variables for next tower (?) #############

    # # Gather summaries from last tower
    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

    # Calculate Gradient
    grad_g, grad_d = compute_gradients((loss_g, loss_d),
                                       (optim_g, optim_d),
                                       var_lists=(var_collections['generator'],
                                                  var_collections['discriminator']))

    return loss_g, loss_d, grad_g, grad_d, image_gen_100


def get_losses(discriminator, vae_sampler,
               num_classes, data_format, ld,
               # images
               images, sketches,
               image_d,
               image_gens, image_gens_b,
               # latent and labels
               image_data_class_id, image_data_class_id_d,
               image_gens_noise, image_gens_noise_b,
               image_labels, image_labels_b,
               # critic out
               real_disc_out, fake_disc_out,
               # logit
               real_logit, fake_logit,
               ):

    def perturb(input_data):
        input_dims = len(input_data.get_shape())
        reduce_axes = [0] + list(range(1, input_dims))
        ret = input_data + 0.5 * tf.sqrt(tf.nn.moments(input_data, axes=reduce_axes)[1]) * tf.random_uniform(input_data.shape)
        # ret = input_data + tf.random_normal(input_data.shape, stddev=2.0)
        return ret

    def get_acgan_loss_focal(real_image_logits_out, real_image_label,
                             disc_image_logits_out, condition,
                             num_classes, ld1=1, ld2=0.5, ld_focal=2.):
        loss_ac_d = tf.reduce_mean((1 - tf.reduce_sum(tf.nn.softmax(real_image_logits_out) * tf.squeeze(
            tf.one_hot(real_image_label, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1)) ** ld_focal *
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label))
        loss_ac_d = ld1 * loss_ac_d

        loss_ac_g = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_image_logits_out, labels=condition))
        loss_ac_g = ld2 * loss_ac_g
        return loss_ac_g, loss_ac_d

    def get_acgan_loss_orig(real_image_logits_out, real_image_label,
                            disc_image_logits_out, condition, num_classes,
                            ld1=1, ld2=0.5):
        loss_ac_d = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label))
        loss_ac_d = ld1 * loss_ac_d

        loss_ac_g = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_image_logits_out, labels=condition))
        loss_ac_g = ld2 * loss_ac_g
        return loss_ac_g, loss_ac_d

    def get_loss_wgan_global_gp(discriminator, data_format,
                                fake_data_out, fake_data_out_, real_data_out,
                                fake_data, real_data,
                                scope=None, ld=ld):
        assert scope is not None
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        assert ndim == 4
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_g = -tf.reduce_mean(fake_data_out_)
        loss_d = tf.reduce_mean(fake_data_out) - tf.reduce_mean(real_data_out)

        # Gradient penalty
        batch_size = int(real_data.get_shape()[0])
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = fake_data - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.add_to_collection("GAN_loss_d_gp", gradient_penalty)

        loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_wgan_sn(discriminator, data_format,
                         fake_data_out, fake_data_out_, real_data_out,
                         fake_data, real_data,
                         scope=None):
        assert scope is not None
        assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        assert ndim == 4
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_g = tf.reduce_mean(tf.nn.softplus(-fake_data_out_))
        loss_d = tf.reduce_mean(tf.nn.softplus(fake_data_out)) + tf.reduce_mean(tf.nn.softplus(-real_data_out))

        # # Gradient penalty
        # batch_size = int(real_data.get_shape()[0])
        # alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
        #                           minval=0., maxval=1., dtype=tf.float32)
        # diff = fake_data - real_data
        # interp = real_data + (alpha * diff)
        # gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
        #                                        data_format=data_format, scope_name=scope)[0],
        #                          [interp])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        # gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
        # tf.add_to_collection("GAN_loss_d_gp", gradient_penalty)
        #
        # loss_d += ld * gradient_penalty

        return loss_g, loss_d

    def get_loss_original_gan_local_gp_one_side_multi(discriminator, data_format,
                                                      fake_data_out, fake_data_out_, real_data_out,
                                                      fake_data, real_data,
                                                      scope=None, ld=ld):
        assert scope is not None
        # assert real_data.get_shape()[0] == fake_data.get_shape()[0]
        ndim = len(real_data.get_shape())
        ndim_out = len(fake_data_out.get_shape())
        assert ndim == 4
        assert ndim_out == 4 or ndim_out == 2
        if ndim_out == 4:
            sum_axis = (1, 2, 3)
        else:
            sum_axis = 1
        if data_format == 'NCHW':
            concat_axis = 1
        else:
            concat_axis = 3

        loss_d_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.zeros_like(fake_data_out)), axis=sum_axis))
        loss_d_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_data_out, labels=tf.ones_like(real_data_out)), axis=sum_axis))
        loss_g_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_data_out, labels=tf.ones_like(fake_data_out)), axis=sum_axis))
        loss_g = loss_g_fake
        loss_d = loss_d_fake + loss_d_real
        loss_d /= 2

        batch_size = int(real_data.get_shape()[0])

        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1] if ndim == 4 else [batch_size, 1],
                                  minval=0., maxval=1., dtype=tf.float32)
        diff = perturb(real_data) - real_data
        interp = real_data + (alpha * diff)
        gradients = tf.gradients(discriminator(interp, num_classes=num_classes, reuse=True,
                                               data_format=data_format, scope_name=scope)[0],
                                 [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3] if ndim == 4 else [1]))
        gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
        loss_d += ld * gradient_penalty
        tf.add_to_collection("GAN_loss_d_gp", gradient_penalty)

        return loss_g, loss_d

    def build_inception(inputs, reuse=True, scope='InceptionV4'):
        is_training = False
        arg_scope = inception_v4_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                    logits, end_points = inception_v4_base(inputs, final_endpoint='Mixed_5b', scope=scope)
        return [end_points['Conv2d_2a_3x3'], end_points['Mixed_4a'], end_points['Mixed_5b']]

    def build_vgg(inputs, reuse=True, scope='vgg_16', num=0):

        def get_endpoint(end_points, name):
            for key in end_points.keys():
                if name in key:
                    return end_points[key]

        is_training = False
        arg_scope = vgg_arg_scope(weight_decay=0.0)
        with slim.arg_scope(arg_scope):
            logits, end_points, my_end_points = vgg_16(inputs, is_training=is_training,
                                                       reuse=reuse, scope=scope, num=num)
        return [get_endpoint(end_points, 'conv1_2'), get_endpoint(end_points, 'conv2_2'),
                get_endpoint(end_points, 'conv3_2'), get_endpoint(end_points, 'conv4_2'), ]

    def get_perceptual_loss(image1, image2, data_format, type="Inception", reuse=True):
        assert data_format == 'NCHW'

        image1 = tf.transpose(image1, (0, 2, 3, 1))
        image2 = tf.transpose(image2, (0, 2, 3, 1))

        if type == "Inception":
            # Normalize to 0-1
            image1 = (image1 + 1) / 2.
            image2 = (image2 + 1) / 2.

            dim = 299

            # Resize to 299, 299
            image1 = tf.image.resize_bilinear(image1, [dim, dim])
            image2 = tf.image.resize_bilinear(image2, [dim, dim])

            image1_lys = build_inception(image1, reuse=reuse)
            image2_lys = build_inception(image2)
        elif type == "vgg":
            image_size = image1.get_shape().as_list()

            dim = 224

            _R_MEAN = tf.constant(123.68, shape=[image_size[0], dim, dim, 1], dtype=tf.float32)
            _G_MEAN = tf.constant(116.78, shape=[image_size[0], dim, dim, 1], dtype=tf.float32)
            _B_MEAN = tf.constant(103.94, shape=[image_size[0], dim, dim, 1], dtype=tf.float32)

            _MEAN = tf.concat([_R_MEAN, _G_MEAN, _B_MEAN], axis=3)

            # Normalize to 0-255
            image1 = (image1 + 1) * 255. / 2.
            image2 = (image2 + 1) * 255. / 2.

            # Resize to 299, 299
            image1 = tf.image.resize_bilinear(image1, [dim, dim])
            image2 = tf.image.resize_bilinear(image2, [dim, dim])

            # Substract mean
            image1 -= _MEAN
            image2 -= _MEAN

            image1_lys = build_vgg(image1, reuse=reuse, num=0)
            image2_lys = build_vgg(image2, num=1)
        else:
            raise ValueError("Network type unknown.")

        tf.add_to_collection("inception_layer_1_1", image1_lys[0])
        tf.add_to_collection("inception_layer_1_2", image1_lys[1])
        tf.add_to_collection("inception_layer_1_3", image1_lys[2])

        tf.add_to_collection("inception_layer_2_1", image2_lys[0])
        tf.add_to_collection("inception_layer_2_2", image2_lys[1])
        tf.add_to_collection("inception_layer_2_3", image2_lys[2])

        loss_perceptual = 0.
        for i in range(len(image2_lys)):
            loss_perceptual += tf.reduce_mean(tf.abs(image2_lys[i] - image1_lys[i]))    # L1
            # loss_perceptual += coeffs[i] * tf.sqrt(tf.reduce_sum(tf.square(image2_lys[i] - image1_lys[i]), axis=[1, 2, 3]))    # L2
            # loss_perceptual = coeffs[i] * models.vae_loss_reconstruct(image2_lys[i], image1_lys[i])       # log-likelihood
        return loss_perceptual

    if Config.sn:
        get_gan_loss = get_loss_wgan_sn
    else:
        if Config.wgan:
            get_gan_loss = get_loss_wgan_global_gp
        else:
            get_gan_loss = get_loss_original_gan_local_gp_one_side_multi

    # GAN Loss, current stage
    loss_g_gan, loss_d_gan = get_gan_loss(discriminator, data_format,
                                          fake_disc_out,
                                          fake_disc_out,
                                          real_disc_out,
                                          image_gens,
                                          image_d,
                                          scope='discriminator')
    tf.add_to_collection("GAN_loss_g", loss_g_gan)
    tf.add_to_collection("GAN_loss_d", loss_d_gan)

    # ACGAN loss
    if not Config.proj_d:
        loss_g_ac, loss_d_ac = get_acgan_loss_focal(real_logit, image_data_class_id_d,
                                                    fake_logit, image_labels,
                                                    num_classes=num_classes)
        tf.add_to_collection("ACGAN_loss_g", loss_g_ac)
        tf.add_to_collection("ACGAN_loss_d", loss_d_ac)

        loss_g_gan += loss_g_ac
        loss_d_gan += loss_d_ac

    # Direct loss
    loss_gt = 0.
    loss_gt += tf.losses.absolute_difference(images, image_gens)  # L1
    loss_gt += 0.3 * get_perceptual_loss(images, image_gens,
                                         data_format=data_format, type="Inception", reuse=False)  # Perceptual
    tf.add_to_collection("direct_loss", loss_gt)

    # Diversity loss
    loss_dv = 0.
    this_loss_dv = tf.abs(image_gens - image_gens_b)  # L1
    this_loss_dv = this_loss_dv / tf.reshape(tf.norm(image_gens_noise - image_gens_noise_b, axis=1), (-1, 1, 1, 1))
    loss_dv -= tf.reduce_mean(this_loss_dv)
    tf.add_to_collection("diversity_loss", loss_dv)

    # Regularization/Weight Decay loss
    loss_decay_g = tf.losses.get_regularization_loss(scope='generator')
    loss_decay_d = tf.losses.get_regularization_loss(scope='discriminator')
    tf.add_to_collection("weight_decay_loss_g", loss_decay_g)
    tf.add_to_collection("weight_decay_loss_d", loss_decay_d)

    coeff_gt = 10
    coeff_dv = 10
    coeff_decay = 1

    loss_g = loss_g_gan + coeff_gt * loss_gt + coeff_dv * loss_dv + coeff_decay * loss_decay_g
    loss_d = loss_d_gan + coeff_decay * loss_decay_d

    tf.add_to_collection("total_loss_g", loss_g)
    tf.add_to_collection("total_loss_d", loss_d)

    return loss_g, loss_d


def get_optimizer(optimizer_name, **kwargs):
    if optimizer_name.lower() == 'RMSProp'.lower():
        return functools.partial(tf.train.RMSPropOptimizer, decay=0.9, momentum=0.0, epsilon=1e-10)
    elif optimizer_name.lower() == 'Adam'.lower():
        return functools.partial(tf.train.AdamOptimizer, beta1=0., beta2=0.9)
        # return functools.partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9)
    elif optimizer_name.lower() == 'AdaDelta'.lower():
        return tf.train.AdadeltaOptimizer
    elif optimizer_name.lower() == 'AdaGrad'.lower():
        return tf.train.AdagradOptimizer


def optimize(gradients, optim, global_step, summaries, global_norm=None, global_norm_clipped=None, appendix=''):
    """Modified from sugartensor"""

    # Add Summary
    if summaries is None:
        summaries = ["loss", "learning_rate"]
    if "gradient_norm" in summaries:
        if global_norm is None:
            tf.summary.scalar("global_norm/gradient_norm" + appendix,
                              clip_ops.global_norm(list(zip(*gradients))[0]))
        else:
            tf.summary.scalar("global_norm/gradient_norm" + appendix,
                              global_norm)
        if global_norm_clipped is not None:
            tf.summary.scalar("global_norm/gradient_norm_clipped" + appendix,
                              global_norm_clipped)

    # Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        if grad_values is not None:
            var_name = variable.name.replace(":", "_")
            if "gradients" in summaries:
                tf.summary.histogram("gradients/%s" % var_name, grad_values)
            if "gradient_norm" in summaries:
                tf.summary.scalar("gradient_norm/%s" % var_name,
                                  clip_ops.global_norm([grad_values]))

    # Gradient Update OP
    return optim.apply_gradients(gradients, global_step=global_step)
