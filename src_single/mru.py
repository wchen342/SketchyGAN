import tensorflow as tf
from tensorflow.python.ops import init_ops
import tensorflow.contrib.layers as ly
import functools

from sn import spectral_normed_weight
from config import Config


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak * x, x)


def mean_pool(input, data_format):
    assert data_format == 'NCHW'
    output = tf.add_n(
        [input[:, :, ::2, ::2], input[:, :, 1::2, ::2], input[:, :, ::2, 1::2], input[:, :, 1::2, 1::2]]) / 4.
    return output


def upsample(input, data_format):
    assert data_format == 'NCHW'
    output = tf.concat([input, input, input, input], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def embed_labels(inputs, num_classes, output_dim, sn,
                 weight_decay_rate=1e-5,
                 reuse=None, scope=None):
    # TODO move regularizer definitions to model
    weights_regularizer = ly.l2_regularizer(weight_decay_rate)

    with tf.variable_scope(scope, 'embedding', [inputs], reuse=reuse) as sc:
        inputs = tf.convert_to_tensor(inputs)

        weights = tf.get_variable(name="weights", shape=(num_classes, output_dim),
                                  initializer=init_ops.random_normal_initializer)

        # Spectral Normalization
        if sn:
            weights = spectral_normed_weight(weights, num_iters=1, update_collection=Config.SPECTRAL_NORM_UPDATE_OPS)

        embed_out = tf.nn.embedding_lookup(weights, inputs)

    return embed_out


def fully_connected(inputs, num_outputs, sn, activation_fn=None,
                    normalizer_fn=None, normalizer_params=None,
                    weights_initializer=ly.xavier_initializer(),
                    weight_decay_rate=1e-6,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None, scope=None):
    # TODO move regularizer definitions to model
    weights_regularizer = ly.l2_regularizer(weight_decay_rate)

    input_dim = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, 'fully_connected', [inputs], reuse=reuse) as sc:
        inputs = tf.convert_to_tensor(inputs)

        weights = tf.get_variable(name="weights", shape=(input_dim, num_outputs),
                                  initializer=weights_initializer, regularizer=weights_regularizer,
                                  trainable=True, dtype=inputs.dtype.base_dtype)

        # Spectral Normalization
        if sn:
            weights = spectral_normed_weight(weights, num_iters=1, update_collection=Config.SPECTRAL_NORM_UPDATE_OPS)

        linear_out = tf.matmul(inputs, weights)

        if biases_initializer is not None:
            biases = tf.get_variable(name="biases", shape=(num_outputs,),
                                     initializer=biases_initializer, regularizer=biases_regularizer,
                                     trainable=True, dtype=inputs.dtype.base_dtype)

        linear_out = tf.nn.bias_add(linear_out, biases)

        # Apply normalizer function / layer.
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            linear_out = normalizer_fn(linear_out, activation_fn=None, **normalizer_params)

        if activation_fn is not None:
            linear_out = activation_fn(linear_out)

    return linear_out


def conv2d(inputs, num_outputs, kernel_size, sn, stride=1, rate=1,
           data_format='NCHW', activation_fn=tf.nn.relu,
           normalizer_fn=None, normalizer_params=None,
           weights_regularizer=None,
           weights_initializer=ly.xavier_initializer(),
           biases_initializer=init_ops.zeros_initializer(),
           biases_regularizer=None,
           reuse=None, scope=None):
    assert data_format == 'NCHW'
    assert rate == 1
    if data_format == 'NCHW':
        channel_axis = 1
        stride = [1, 1, stride, stride]
        rate = [1, 1, rate, rate]
    else:
        channel_axis = 3
        stride = [1, stride, stride, 1]
        rate = [1, rate, rate, 1]
    input_dim = inputs.get_shape().as_list()[channel_axis]

    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse) as sc:
        inputs = tf.convert_to_tensor(inputs)

        weights = tf.get_variable(name="weights", shape=(kernel_size, kernel_size, input_dim, num_outputs),
                                  initializer=weights_initializer, regularizer=weights_regularizer,
                                  trainable=True, dtype=inputs.dtype.base_dtype)
        # Spectral Normalization
        if sn:
            weights = spectral_normed_weight(weights, num_iters=1, update_collection=Config.SPECTRAL_NORM_UPDATE_OPS)

        conv_out = tf.nn.conv2d(inputs, weights, strides=stride, padding='SAME', data_format=data_format)

        if biases_initializer is not None:
            biases = tf.get_variable(name='biases', shape=(1, num_outputs, 1, 1),
                                     initializer=biases_initializer, regularizer=biases_regularizer,
                                     trainable=True, dtype=inputs.dtype.base_dtype)
            conv_out += biases

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            conv_out = normalizer_fn(conv_out, activation_fn=None, **normalizer_params)

        if activation_fn is not None:
            conv_out = activation_fn(conv_out)

    return conv_out


def conv_mean_pool(inputs, num_outputs, kernel_size, sn, rate=1,
                   activation_fn=None,
                   normalizer_fn=None, normalizer_params=None,
                   weights_regularizer=None,
                   weights_initializer=ly.xavier_initializer_conv2d(),
                   biases_initializer=tf.zeros_initializer(),
                   data_format='NCHW'):
    output = conv2d(inputs, num_outputs, kernel_size, sn=sn, rate=rate, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer,
                    data_format=data_format)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def mean_pool_conv(inputs, num_outputs, kernel_size, sn, rate=1,
                   activation_fn=None,
                   normalizer_fn=None, normalizer_params=None,
                   weights_regularizer=None,
                   weights_initializer=ly.xavier_initializer_conv2d(),
                   data_format='NCHW'):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = conv2d(output, num_outputs, kernel_size, sn=sn, rate=rate, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                    data_format=data_format)
    return output


def upsample_conv(inputs, num_outputs, kernel_size, sn, activation_fn=None,
                  normalizer_fn=None, normalizer_params=None,
                  weights_regularizer=None,
                  weights_initializer=ly.xavier_initializer_conv2d(),
                  biases_initializer=tf.zeros_initializer(),
                  data_format='NCHW'):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1 if data_format == 'NCHW' else 3)
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 3, 1, 2])
    output = conv2d(output, num_outputs, kernel_size, sn=sn, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer,
                    data_format=data_format)
    return output


def upsample_conv_bilinear(inputs, num_outputs, kernel_size, sn, activation_fn=None,
                           normalizer_fn=None, normalizer_params=None,
                           weights_regularizer=None,
                           weights_initializer=ly.xavier_initializer_conv2d(),
                           data_format='NCHW'):
    output = inputs
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 2, 3, 1])
    batch_size, height, width, channel = [int(i) for i in output.get_shape()]
    # output = tf.Print(output, [tf.reduce_min(output), tf.reduce_max(output)], message='before')
    output = tf.image.resize_bilinear(output, [height * 2, width * 2])
    # output = tf.Print(output, [tf.reduce_min(output), tf.reduce_max(output)], message='after')
    slice0 = output[:, :, :, 0::4]
    slice1 = output[:, :, :, 1::4]
    slice2 = output[:, :, :, 2::4]
    slice3 = output[:, :, :, 3::4]
    output = slice0 + slice1 + slice2 + slice3
    if data_format == 'NCHW':
        output = tf.transpose(output, [0, 3, 1, 2])
    output = conv2d(output, num_outputs, kernel_size, sn=sn, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_regularizer=weights_regularizer, weights_initializer=weights_initializer,
                    data_format=data_format)
    return output


# Sigmoid Gates
def mru_conv_block(inp, ht, filter_depth, sn,
                   stride, dilate=1,
                   activation_fn=tf.nn.relu,
                   normalizer_fn=None,
                   normalizer_params=None,
                   weights_initializer=ly.xavier_initializer_conv2d(),
                   data_format='NCHW',
                   weight_decay_rate=1e-8, norm_mask=False):
    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)
    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    full_inp = tf.concat([ht, inp], axis=channel_index)

    # update gate
    rg = conv2d(full_inp, ht.get_shape().as_list()[channel_index], 3, sn=sn, stride=1, rate=dilate,
                data_format=data_format, activation_fn=tf.nn.sigmoid,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)
    # output gate
    zg = conv2d(full_inp, filter_depth, 3, sn=sn, stride=1, rate=dilate, data_format=data_format,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)

    # new hidden state
    h_new = conv2d(tf.concat([rg * ht, inp], axis=channel_index), filter_depth, 3, sn=sn, stride=1, rate=dilate,
                   data_format=data_format, activation_fn=activation_fn,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)
    h_new = conv2d(h_new, filter_depth, 3, sn=sn, stride=1, rate=dilate,
                   data_format=data_format, activation_fn=activation_fn,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)

    # new hidden state out
    # projection for filter depth
    if ht.get_shape().as_list()[channel_index] != filter_depth:
        ht = conv2d(ht, filter_depth, 1, sn=sn, stride=1, data_format=data_format, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_regularizer=regularizer,
                    weights_initializer=weights_initializer)
    ht_new = ht * (1 - zg) + h_new * zg

    if stride == 2:
        ht_new = mean_pool(ht_new, data_format=data_format)
    elif stride != 1:
        raise NotImplementedError

    return ht_new


# LReLU Gates
def mru_conv_block_v2(inp, ht, filter_depth, sn,
                      stride, dilate=1,
                      activation_fn=tf.nn.relu,
                      normalizer_fn=None,
                      normalizer_params=None,
                      weights_initializer=ly.xavier_initializer_conv2d(),
                      data_format='NCHW',
                      weight_decay_rate=1e-8, norm_mask=False):
    channel_index = 1 if data_format == 'NCHW' else 3
    reduce_dim = [2, 3] if data_format == 'NCHW' else [1, 2]
    regularizer = ly.l2_regularizer(weight_decay_rate)
    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    full_inp = tf.concat([ht, inp], axis=channel_index)

    # update gate
    rg = conv2d(full_inp, ht.get_shape().as_list()[channel_index], 3, sn=sn, stride=1, rate=dilate,
                data_format=data_format, activation_fn=lrelu,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)
    rg = (rg - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True)) / (
            tf.reduce_max(rg, axis=reduce_dim, keep_dims=True) - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True))
    # output gate
    zg = conv2d(full_inp, filter_depth, 3, sn=sn, stride=1, rate=dilate, data_format=data_format,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                activation_fn=lrelu,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)
    zg = (zg - tf.reduce_min(zg, axis=reduce_dim, keep_dims=True)) / (
            tf.reduce_max(zg, axis=reduce_dim, keep_dims=True) - tf.reduce_min(zg, axis=reduce_dim, keep_dims=True))

    # new hidden state
    h_new = conv2d(tf.concat([rg * ht, inp], axis=channel_index), filter_depth, 3, sn=sn, stride=1, rate=dilate,
                   data_format=data_format, activation_fn=activation_fn,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)
    h_new = conv2d(h_new, filter_depth, 3, sn=sn, stride=1, rate=dilate,
                   data_format=data_format, activation_fn=activation_fn,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)

    # new hidden state out
    # projection for filter depth
    if ht.get_shape().as_list()[channel_index] != filter_depth:
        ht = conv2d(ht, filter_depth, 1, sn=sn, stride=1, data_format=data_format, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_regularizer=regularizer,
                    weights_initializer=weights_initializer)
    ht_new = ht * (1 - zg) + h_new * zg

    if stride == 2:
        ht_new = mean_pool(ht_new, data_format=data_format)
    elif stride != 1:
        raise NotImplementedError

    return ht_new


# No output gates
def mru_conv_block_v3(inp, ht, filter_depth, sn,
                      stride, dilate=1,
                      activation_fn=tf.nn.relu,
                      normalizer_fn=None,
                      normalizer_params=None,
                      weights_initializer=ly.xavier_initializer_conv2d(),
                      biases_initializer_mask=tf.constant_initializer(value=0.5),
                      biases_initializer_h=tf.constant_initializer(value=-1),
                      data_format='NCHW',
                      weight_decay_rate=1e-8,
                      norm_mask=False,
                      norm_input=True,
                      deconv=False):

    def norm_activ(tensor_in):
        if normalizer_fn is not None:
            _normalizer_params = normalizer_params or {}
            tensor_normed = normalizer_fn(tensor_in, **_normalizer_params)
        else:
            tensor_normed = tf.identity(tensor_in)
        if activation_fn is not None:
            tensor_normed = activation_fn(tensor_normed)

        return tensor_normed

    channel_index = 1 if data_format == 'NCHW' else 3
    reduce_dim = [2, 3] if data_format == 'NCHW' else [1, 2]
    hidden_depth = ht.get_shape().as_list()[channel_index]
    regularizer = ly.l2_regularizer(weight_decay_rate) if weight_decay_rate > 0 else None
    weights_initializer_mask = weights_initializer
    biases_initializer = tf.zeros_initializer()

    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    if deconv:
        if stride == 2:
            ht = upsample(ht, data_format=data_format)
        elif stride != 1:
            raise NotImplementedError

    ht_orig = tf.identity(ht)

    # Normalize hidden state
    with tf.variable_scope('norm_activation_in') as sc:
        if norm_input:
            full_inp = tf.concat([norm_activ(ht), inp], axis=channel_index)
        else:
            full_inp = tf.concat([ht, inp], axis=channel_index)

    # update gate
    rg = conv2d(full_inp, hidden_depth, 3, sn=sn, stride=1, rate=dilate,
                data_format=data_format, activation_fn=lrelu,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer_mask,
                biases_initializer=biases_initializer_mask,
                scope='update_gate')
    rg = (rg - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True)) / (
            tf.reduce_max(rg, axis=reduce_dim, keep_dims=True) - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True))

    # Input Image conv
    img_new = conv2d(inp, hidden_depth, 3, sn=sn, stride=1, rate=dilate,
                     data_format=data_format, activation_fn=None,
                     normalizer_fn=None, normalizer_params=None,
                     biases_initializer=biases_initializer,
                     weights_regularizer=regularizer,
                     weights_initializer=weights_initializer)

    ht_plus = ht + rg * img_new
    with tf.variable_scope('norm_activation_merge_1') as sc:
        ht_new_in = norm_activ(ht_plus)

    # new hidden state
    h_new = conv2d(ht_new_in, filter_depth, 3, sn=sn, stride=1, rate=dilate,
                   data_format=data_format, activation_fn=activation_fn,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   biases_initializer=biases_initializer,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)
    h_new = conv2d(h_new, filter_depth, 3, sn=sn, stride=1, rate=dilate,
                   data_format=data_format, activation_fn=None,
                   normalizer_fn=None, normalizer_params=None,
                   biases_initializer=biases_initializer,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)

    # new hidden state out
    # linear project for filter depth
    if ht.get_shape().as_list()[channel_index] != filter_depth:
        ht_orig = conv2d(ht_orig, filter_depth, 1, sn=sn, stride=1,
                         data_format=data_format, activation_fn=None,
                         normalizer_fn=None, normalizer_params=None,
                         biases_initializer=biases_initializer,
                         weights_regularizer=regularizer,
                         weights_initializer=weights_initializer)
    ht_new = ht_orig + h_new

    if not deconv:
        if stride == 2:
            ht_new = mean_pool(ht_new, data_format=data_format)
        elif stride != 1:
            raise NotImplementedError

    return ht_new


# Sigmoid gates
def mru_deconv_block(inp, ht, filter_depth, sn, stride,
                     activation_fn=tf.nn.relu,
                     normalizer_fn=None,
                     normalizer_params=None,
                     weights_initializer=ly.xavier_initializer_conv2d(),
                     data_format='NCHW',
                     weight_decay_rate=1e-8,
                     norm_mask=False):
    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)

    if stride == 2:
        ht = upsample(ht, data_format=data_format)
    elif stride != 1:
        raise NotImplementedError

    full_inp = tf.concat([ht, inp], axis=channel_index)

    # update gate
    rg = conv2d(full_inp, ht.get_shape().as_list()[channel_index], 3, sn=sn, stride=1, data_format=data_format,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)
    # output gate
    zg = conv2d(full_inp, filter_depth, 3, sn=sn, stride=1, data_format=data_format,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                activation_fn=tf.nn.sigmoid,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)

    # new hidden state
    h_new = conv2d(tf.concat([rg * ht, inp], axis=channel_index), filter_depth, 3, sn=sn, stride=1, data_format=data_format,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   activation_fn=activation_fn,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)
    h_new = conv2d(h_new, filter_depth, 3, sn=sn, stride=1, data_format=data_format,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   activation_fn=activation_fn,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)

    # new hidden state out
    # projection for filter depth
    if ht.get_shape().as_list()[channel_index] != filter_depth:
        ht = conv2d(ht, filter_depth, 1, sn=sn, stride=1, data_format=data_format, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_initializer=weights_initializer)
    ht_new = ht * (1 - zg) + h_new * zg

    return ht_new


# LReLU gates
def mru_deconv_block_v2(inp, ht, filter_depth, sn, stride,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=ly.xavier_initializer_conv2d(),
                        data_format='NCHW',
                        weight_decay_rate=1e-8,
                        norm_mask=False):
    channel_index = 1 if data_format == 'NCHW' else 3
    reduce_dim = [2, 3] if data_format == 'NCHW' else [1, 2]
    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    channel_index = 1 if data_format == 'NCHW' else 3
    regularizer = ly.l2_regularizer(weight_decay_rate)

    if stride == 2:
        ht = upsample(ht, data_format=data_format)
    elif stride != 1:
        raise NotImplementedError

    full_inp = tf.concat([ht, inp], axis=channel_index)

    # update gate
    rg = conv2d(full_inp, ht.get_shape().as_list()[channel_index], 3, sn=sn, stride=1, data_format=data_format,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                activation_fn=lrelu,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)
    rg = (rg - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True)) / (
            tf.reduce_max(rg, axis=reduce_dim, keep_dims=True) - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True))
    # output gate
    zg = conv2d(full_inp, filter_depth, 3, sn=sn, stride=1, data_format=data_format,
                normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                activation_fn=lrelu,
                weights_regularizer=regularizer,
                weights_initializer=weights_initializer)
    zg = (zg - tf.reduce_min(zg, axis=reduce_dim, keep_dims=True)) / (
            tf.reduce_max(zg, axis=reduce_dim, keep_dims=True) - tf.reduce_min(zg, axis=reduce_dim, keep_dims=True))

    # new hidden state
    h_new = conv2d(tf.concat([rg * ht, inp], axis=channel_index), filter_depth, 3, sn=sn, stride=1, data_format=data_format,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   activation_fn=activation_fn,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)
    h_new = conv2d(h_new, filter_depth, 3, sn=sn, stride=1, data_format=data_format,
                   normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                   activation_fn=activation_fn,
                   weights_regularizer=regularizer,
                   weights_initializer=weights_initializer)

    # new hidden state out
    # projection for filter depth
    if ht.get_shape().as_list()[channel_index] != filter_depth:
        ht = conv2d(ht, filter_depth, 1, sn=sn, stride=1, data_format=data_format, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    weights_initializer=weights_initializer)
    ht_new = ht * (1 - zg) + h_new * zg

    return ht_new


def mru_conv(x, ht, filter_depth, sn, stride=2, dilate_rate=1,
             num_blocks=5, last_unit=False,
             activation_fn=tf.nn.relu,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=ly.xavier_initializer_conv2d(),
             weight_decay_rate=1e-5,
             unit_num=0, data_format='NCHW'):
    assert len(ht) == num_blocks

    def norm_activ(tensor_in):
        if normalizer_fn is not None:
            _normalizer_params = normalizer_params or {}
            tensor_normed = normalizer_fn(tensor_in, **_normalizer_params)
        else:
            tensor_normed = tf.identity(tensor_in)
        if activation_fn is not None:
            tensor_normed = activation_fn(tensor_normed)

        return tensor_normed

    if dilate_rate != 1:
        stride = 1

    # cell_block = mru_conv_block
    # cell_block = mru_conv_block_v2
    cell_block = functools.partial(mru_conv_block_v3, deconv=False)

    hts_new = []
    inp = x
    with tf.variable_scope('mru_conv_unit_t_%d_layer_0' % unit_num):
        ht_new = cell_block(inp, ht[0], filter_depth, sn=sn, stride=stride,
                            dilate=dilate_rate,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_initializer=weights_initializer,
                            data_format=data_format,
                            weight_decay_rate=weight_decay_rate)
        hts_new.append(ht_new)
        inp = ht_new

    for i in range(1, num_blocks):
        if stride == 2:
            ht[i] = mean_pool(ht[i], data_format=data_format)
        with tf.variable_scope('mru_conv_unit_t_%d_layer_%d' % (unit_num, i)):
            ht_new = cell_block(inp, ht[i], filter_depth, sn=sn, stride=1,
                                dilate=dilate_rate,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                weights_initializer=weights_initializer,
                                data_format=data_format,
                                weight_decay_rate=weight_decay_rate)
            hts_new.append(ht_new)
            inp = ht_new

    if hasattr(cell_block, 'func') and cell_block.func == mru_conv_block_v3 and last_unit:
        with tf.variable_scope('mru_conv_unit_last_norm'):
            hts_new[-1] = norm_activ(hts_new[-1])

    return hts_new


def mru_deconv(x, ht, filter_depth, sn, stride=2, num_blocks=2,
               last_unit=False,
               activation_fn=tf.nn.relu,
               normalizer_fn=None,
               normalizer_params=None,
               weights_initializer=ly.xavier_initializer_conv2d(),
               weight_decay_rate=1e-5,
               unit_num=0, data_format='NCHW'):
    assert len(ht) == num_blocks

    def norm_activ(tensor_in):
        if normalizer_fn is not None:
            _normalizer_params = normalizer_params or {}
            tensor_normed = normalizer_fn(tensor_in, **_normalizer_params)
        else:
            tensor_normed = tf.identity(tensor_in)
        if activation_fn is not None:
            tensor_normed = activation_fn(tensor_normed)

        return tensor_normed

    # cell_block = mru_deconv_block
    cell_block = mru_deconv_block_v2

    hts_new = []
    inp = x
    with tf.variable_scope('mru_deconv_unit_t_%d_layer_0' % unit_num):
        ht_new = cell_block(inp, ht[0], filter_depth, sn=sn, stride=stride,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_initializer=weights_initializer,
                            data_format=data_format,
                            weight_decay_rate=weight_decay_rate)
        hts_new.append(ht_new)
        inp = ht_new

    for i in range(1, num_blocks):
        if stride == 2:
            ht[i] = upsample(ht[i], data_format=data_format)
        with tf.variable_scope('mru_deconv_unit_t_%d_layer_%d' % (unit_num, i)):
            ht_new = cell_block(inp, ht[i], filter_depth, sn=sn, stride=1,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                weights_initializer=weights_initializer,
                                data_format=data_format,
                                weight_decay_rate=weight_decay_rate)
            hts_new.append(ht_new)
            inp = ht_new

    # if last_unit:
    #     with tf.variable_scope('mru_deconv_unit_last_norm'):
    #         hts_new[-1] = norm_activ(hts_new[-1])

    return hts_new
