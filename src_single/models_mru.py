import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly

from mru import embed_labels, fully_connected, conv2d, mean_pool, upsample, mru_conv, mru_deconv
from config import Config

SIZE = 64
NUM_BLOCKS = 1


def image_resize(inputs, size, method, data_format):
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    out = tf.image.resize_images(inputs, size, method)
    if data_format == 'NCHW':
        out = tf.transpose(out, [0, 3, 1, 2])
    return out


def batchnorm(inputs, data_format=None, activation_fn=None, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if data_format != 'NCHW':
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, (0, 2, 3) if len(inputs.shape) == 4 else (0,), keep_dims=True)
    shape = mean.get_shape().as_list()  # shape is [1,n,1,1]
    offset_m = tf.get_variable('offset', initializer=np.zeros([n_labels, shape[1]], dtype='float32'))
    scale_m = tf.get_variable('scale', initializer=np.ones([n_labels, shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var,
                                       offset[:, :, None, None] if len(inputs.shape) == 4 else offset[:, :],
                                       scale[:, :, None, None] if len(inputs.shape) == 4 else scale[:, :],
                                       1e-5)
    return result


def conditional(inputs, data_format=None, activation_fn=None, labels=None, n_labels=10):
    if data_format != 'NCHW':
        raise Exception('unsupported')
    with tf.variable_scope(None, 'conditional_shift'):
        depth = inputs.get_shape().as_list()[1]
        offset_m = tf.get_variable('offset', initializer=np.zeros([n_labels, depth], dtype='float32'))
        scale_m = tf.get_variable('scale', initializer=np.ones([n_labels, depth], dtype='float32'))
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)
        result = inputs * scale[:, :, None, None] + offset[:, :, None, None]
    return result


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak * x, x)


def prelu(x, name="prelu"):
    with tf.variable_scope(name):
        leak = tf.get_variable("param", shape=None, initializer=0.2, regularizer=None,
                               trainable=True, caching_device=None)
        return tf.maximum(leak * x, x)


def miu_relu(x, miu=0.7, name="miu_relu"):
    with tf.variable_scope(name):
        return (x + tf.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def image_encoder_mru(x, num_classes, reuse=False, data_format='NCHW', labels=None, scope_name=None):
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear
    sn = False

    if normalizer_params_e is not None and normalizer_fn_e != ly.batch_norm and normalizer_fn_e != ly.layer_norm:
        normalizer_params_e['labels'] = labels
        normalizer_params_e['n_labels'] = num_classes

    if data_format == 'NCHW':
        x_list = []
        resized_ = x
        x_list.append(resized_)

        for i in range(4):
            resized_ = mean_pool(resized_, data_format=data_format)
            x_list.append(resized_)
        x_list = x_list[::-1]
    else:
        raise NotImplementedError

    output_list = []

    h0 = conv2d(x_list[-1], 8, kernel_size=7, sn=sn, stride=2, data_format=data_format,
                activation_fn=None,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=weight_initializer)

    output_list.append(h0)

    # Initial memory state
    hidden_state_shape = h0.get_shape().as_list()
    hidden_state_shape[0] = 1
    hts_0 = [h0]

    hts_1 = mru_conv(x_list[-2], hts_0,
                     size * 1, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=False,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=1)
    output_list.append(hts_1[-1])
    hts_2 = mru_conv(x_list[-3], hts_1,
                     size * 2, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=False,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=2)
    output_list.append(hts_2[-1])
    hts_3 = mru_conv(x_list[-4], hts_2,
                     size * 4, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=False,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=3)
    output_list.append(hts_3[-1])
    hts_4 = mru_conv(x_list[-5], hts_3,
                     size * 8, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=True,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=4)
    output_list.append(hts_4[-1])

    return output_list


def generator_skip(z, output_channel, num_classes, reuse=False, data_format='NCHW',
                   labels=None, scope_name=None):
    print("G")
    size = SIZE
    num_blocks = NUM_BLOCKS
    sn = False

    input_dims = z.get_shape().as_list()
    resize_method = tf.image.ResizeMethod.AREA

    if data_format == 'NCHW':
        height = input_dims[2]
        width = input_dims[3]
    else:
        height = input_dims[1]
        width = input_dims[2]
    resized_z = [tf.identity(z)]
    for i in range(5):
        resized_z.append(image_resize(z, [int(height / 2 ** (i + 1)), int(width / 2 ** (i + 1))],
                                      resize_method, data_format))
    resized_z = resized_z[::-1]

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3

    if normalizer_params_g is not None and normalizer_fn_g != ly.batch_norm and normalizer_fn_g != ly.layer_norm:
        normalizer_params_g['labels'] = labels
        normalizer_params_g['n_labels'] = num_classes

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        z_encoded = image_encoder_mru(z, num_classes=num_classes, reuse=reuse, data_format=data_format,
                                      labels=labels, scope_name=scope_name)

        input_e_dims = z_encoded[-1].get_shape().as_list()
        batch_size = input_e_dims[0]
        channel_depth = int(input_e_dims[concat_axis] / 8.)
        if data_format == 'NCHW':
            noise_dims = [batch_size, channel_depth, int(input_e_dims[2] * 2), int(input_e_dims[3] * 2)]
        else:
            noise_dims = [batch_size, int(input_e_dims[1] * 2), int(input_e_dims[2] * 2), channel_depth]

        noise_vec = tf.random_normal(shape=(batch_size, 256), dtype=tf.float32)
        noise = fully_connected(noise_vec, int(np.prod(noise_dims[1:])), sn=sn,
                                activation_fn=activation_fn_g,
                                normalizer_fn=normalizer_fn_g,
                                normalizer_params=normalizer_params_g)
        noise = tf.reshape(noise, shape=noise_dims)

        # Initial memory state
        hidden_state_shape = z_encoded[-1].get_shape().as_list()
        hidden_state_shape[0] = 1
        hts_0 = [z_encoded[-1]]

        input_0 = tf.concat([resized_z[1], noise], axis=concat_axis)
        hts_1 = mru_deconv(input_0, hts_0,
                           size * 6, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=0)
        input_1 = tf.concat([resized_z[2], z_encoded[-3]], axis=concat_axis)
        hts_2 = mru_deconv(input_1, hts_1,
                           size * 4, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=2)
        input_2 = tf.concat([resized_z[3], z_encoded[-4]], axis=concat_axis)
        hts_3 = mru_deconv(input_2, hts_2,
                           size * 2, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=4)
        input_3 = tf.concat([resized_z[4], z_encoded[-5]], axis=concat_axis)
        hts_4 = mru_deconv(input_3, hts_3,
                           size * 2, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=6)
        hts_5 = mru_deconv(resized_z[5], hts_4,
                           size * 1, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=True,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=8)
        out = conv2d(hts_5[-1], 3, 7, sn=sn, stride=1, data_format=data_format,
                     normalizer_fn=None, activation_fn=tf.nn.tanh,
                     weights_initializer=weight_initializer)
        assert out.get_shape().as_list()[2] == 64
        return out, noise_vec


# MRU
def critic_multiple_proj(x, num_classes, labels=None, reuse=False, data_format='NCHW', scope_name=None):
    print("D")
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear
    sn = Config.sn

    if data_format == 'NCHW':
        channel_axis = 1
    else:
        channel_axis = 3
    if type(x) is list:
        x = x[-1]

    if data_format == 'NCHW':
        x_list = []
        resized_ = x
        x_list.append(resized_)

        for i in range(5):
            resized_ = mean_pool(resized_, data_format=data_format)
            x_list.append(resized_)
        x_list = x_list[::-1]
    else:
        raise NotImplementedError

    output_dim = 1

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        h0 = conv2d(x_list[-1], 8, kernel_size=7, sn=sn, stride=1, data_format=data_format,
                    activation_fn=activation_fn_d,
                    normalizer_fn=normalizer_fn_d,
                    normalizer_params=normalizer_params_d,
                    weights_initializer=weight_initializer)

        # Initial memory state
        hidden_state_shape = h0.get_shape().as_list()
        batch_size = hidden_state_shape[0]
        hidden_state_shape[0] = 1
        hts_0 = [h0]
        for i in range(1, num_blocks):
            h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                         initializer=tf.zeros_initializer()), [batch_size, 1, 1, 1])
            hts_0.append(h0)

        hts_1 = mru_conv(x_list[-1], hts_0,
                         size * 2, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=False,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=1)
        hts_2 = mru_conv(x_list[-2], hts_1,
                         size * 4, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=False,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=2)
        hts_3 = mru_conv(x_list[-3], hts_2,
                         size * 8, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=False,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=3)
        hts_4 = mru_conv(x_list[-4], hts_3,
                         size * 12, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=True,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=4)

        img = hts_4[-1]
        img_shape = img.get_shape().as_list()

        # discriminator end
        disc = conv2d(img, output_dim, kernel_size=1, sn=sn, stride=1, data_format=data_format,
                      activation_fn=None, normalizer_fn=None,
                      weights_initializer=weight_initializer)

        if Config.proj_d:
            # Projection discriminator
            assert labels is not None and (len(labels.get_shape()) == 1 or labels.get_shape().as_list()[-1] == 1)

            class_embeddings = embed_labels(labels, num_classes, img_shape[channel_axis], sn=sn)
            class_embeddings = tf.reshape(class_embeddings, (img_shape[0], img_shape[channel_axis], 1, 1))  # NCHW

            disc += tf.reduce_sum(img * class_embeddings, axis=1, keep_dims=True)

            logits = None
        else:
            # classification end
            img = tf.reduce_mean(img, axis=(2, 3) if data_format == 'NCHW' else (1, 2))
            logits = fully_connected(img, num_classes, sn=sn, activation_fn=None, normalizer_fn=None)

    return disc, logits


weight_initializer = tf.random_normal_initializer(0, 0.02)
# weight_initializer = ly.xavier_initializer_conv2d()


def set_param(data_format='NCHW'):
    global model_data_format, normalizer_fn_e, normalizer_fn_g, normalizer_fn_d, \
        normalizer_params_e, normalizer_params_g, normalizer_params_d
    model_data_format = data_format
    normalizer_fn_e = batchnorm
    normalizer_params_e = {'data_format': model_data_format}
    normalizer_fn_g = batchnorm
    normalizer_params_g = {'data_format': model_data_format}
    normalizer_fn_d = None
    normalizer_params_d = None


model_data_format = None

normalizer_fn_e = None
normalizer_params_e = None
normalizer_fn_g = None
normalizer_params_g = None
normalizer_fn_d = None
normalizer_params_d = None

activation_fn_e = miu_relu
activation_fn_g = miu_relu
activation_fn_d = prelu

generator = generator_skip
critic = critic_multiple_proj
