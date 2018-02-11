import tensorflow as tf
import numpy as np
from . import voxel_dcn


"""
This module provides some short functions to reduce code volume
"""


def voxel_dcl(inputs, out_num, kernel_size, scope, data_type='2D', action='add'):
    outs = voxel_dcn.voxel_dcl3d(inputs, out_num, kernel_size, scope, action, None)
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def ivoxel_dcl(inputs, out_num, kernel_size, scope, data_type='3D', action='add'):
    outs = voxel_dcn.ivoxel_dcl(
            inputs, out_num, kernel_size, scope, action, None)
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def conv(inputs, out_num, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        outs = tf.layers.conv2d(
            inputs, out_num, kernel_size, padding='same', name=scope+'/conv',
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [inputs.shape[-1].value, out_num]
        weights = tf.get_variable(
            scope+'/conv/weights', shape,
            initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d(
            inputs, weights, (1, 1, 1, 1, 1), padding='SAME',
            name=scope+'/conv')
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def deconv(inputs, out_num, kernel_size, scope, data_type='2D', **kws):
    if data_type == '2D':
        outs = tf.layers.conv2d_transpose(
            inputs, out_num, kernel_size, (2, 2), padding='same', name=scope,
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [out_num, out_num]
        input_shape = inputs.shape.as_list()
        out_shape = [input_shape[0]] + \
            list(map(lambda x: x*2, input_shape[1:-1])) + [out_num]
        weights = tf.get_variable(
            scope+'/deconv/weights', shape,
            initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d_transpose(
            inputs, weights, out_shape, (1, 2, 2, 2, 1), name=scope+'/deconv')
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def pool(inputs, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        return tf.layers.max_pooling2d(inputs, kernel_size, (2, 2), name=scope)
    return tf.layers.max_pooling3d(inputs, kernel_size, (2, 2, 2), name=scope)


def dice_ratio(preds, labels):
    '''
    preds & labels should only contain 0 or 1.
    '''
    return np.sum(preds[labels==1])*2.0 / (np.sum(preds) + np.sum(labels))


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)



