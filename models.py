from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim

conv = functools.partial(slim.conv2d, activation_fn=None)
deconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
lrelu = functools.partial(ops.leak_relu, leak=0.2)


def discriminator(img, scope, df_dim=64, reuse=False, train=True):
    """

    Args:
        img:
        scope:
        df_dim:
        reuse:
        train:

    Returns:

    """
    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        # h0: (128x128xdf_dim); h1: (64x64xdf_dim*2); h2: (32x32xdf_dim*4); h3: (32x32xdf_dim*8); pred: (32x32x1).
        h0 = lrelu(conv(img, num_outputs=df_dim, kernel_size=4, stride=2, scope='h0_conv'))
        h1 = lrelu(bn(conv(h0, num_outputs=df_dim * 2, kernel_size=4, stride=2, scope='h1_conv'), scope='h1_bn'))
        h2 = lrelu(bn(conv(h1, num_outputs=df_dim * 4, kernel_size=4, stride=2, scope='h2_conv'), scope='h2_bn'))
        h3 = lrelu(bn(conv(h2, num_outputs=df_dim * 8, kernel_size=4, stride=1, scope='h3_conv'), scope='h3_bn'))
        h4 = conv(h3, num_outputs=1, kernel_size=4, stride=1, scope='h4_conv')
        return h4  # prediction layer (b/c num_outputs=1, hence real or fake).


def generator(img, scope, gf_dim=64, n_res_blocks=9, reuse=False, train=True):
    """

    Args:
        img:
        scope:
        gf_dim:
        n_res_blocks:
        reuse:
        train:

    Returns:

    """
    if not isinstance(n_res_blocks, int) or not n_res_blocks > 1:
        raise ValueError("`n_res_blocks` must be an integer greater than 1.")

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    def residule_block(x, dim, scope='res'):
        y = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        y = relu(bn(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv1'), scope=scope + '_bn1'))
        y = tf.pad(y, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        y = bn(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv2'), scope=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        # Encode
        c0 = tf.pad(img, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        c1 = relu(bn(conv(c0, num_outputs=gf_dim, kernel_size=7, stride=1, padding='VALID', scope='c1_conv'), scope='c1_bn'))
        c2 = relu(bn(conv(c1, num_outputs=gf_dim * 2, kernel_size=3, stride=2, scope='c2_conv'), scope='c2_bn'))
        c3 = relu(bn(conv(c2, num_outputs=gf_dim * 4, kernel_size=3, stride=2, scope='c3_conv'), scope='c3_bn'))

        # Transform
        r = residule_block(c3, gf_dim * 4, scope='r1')
        for i in range(2, n_res_blocks + 1):
            r = residule_block(r, gf_dim * 4, scope='r{0}'.format(str(i)))

        # Decode
        d1 = relu(bn(deconv(r, num_outputs=gf_dim * 2, kernel_size=3, stride=2, scope='d1_dconv'), scope='d1_bn'))
        d2 = relu(bn(deconv(d1, num_outputs=gf_dim, kernel_size=3, stride=2, scope='d2_dconv'), scope='d2_bn'))
        d2 = tf.pad(d2, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        pred = conv(d2, num_outputs=3, kernel_size=7, stride=1, padding='VALID', scope='pred_conv')

        # Add Activation Function
        pred = tf.nn.tanh(pred)

        return pred
