import tensorflow as tf
import numpy as np


"""
This module realizes the three methods proposed in paper
[Pixel Deconvolutional Networks] (https://arxiv.org/abs/1705.06820)

pixel_dcl: realizes Pixel Deconvolutional Layer
ipixel_dcl: realizes Input Pixel Deconvolutional Layer
ipixel_dcl: realizes Input Pixel Convolutional Layer
"""


def ivoxel_dcl(inputs, out_num, kernel_size, scope, action='concat', activation_fn=tf.nn.relu):
    """
    inputs: input tensor
    out_num: output channel number
    kernel_size: convolutional kernel size
    scope: operation scope
    activation_fn: activation function, could be None if needed
    """
    axis, c_axis = (1, 2, 3), 4  # only support format 'NDHWC'
    conv0 = conv3d(inputs, out_num, kernel_size, scope+'/conv0')
    combine1 = combine([inputs, conv0], action, c_axis, scope+'combine1')
    conv1 = conv3d(combine1, out_num, kernel_size, scope+'/conv1')
    combine2 = combine([combine1, conv1], action, c_axis, scope+'/combine2')
    conv2 = conv3d(combine2, 3*out_num, kernel_size, scope+'/conv2')
    conv2_list = tf.split(conv2, 3, c_axis, name=scope+'/split1')
    combine3 = combine(conv2_list+[combine2], action, c_axis, scope+'/combine3')
    conv3 = conv3d(combine3, 3*out_num, kernel_size, scope+'/conv3')
    conv3_list = tf.split(conv3, 3, c_axis, name=scope+'/split2')
    dilated_conv0 = dilate_tensor(
        conv0, axis, (0, 0, 0), scope+'/dialte_conv0')
    dilated_conv1 = dilate_tensor(
        conv1, axis, (1, 1, 1), scope+'/dialte_conv1')
    dilated_list = [dilated_conv0, dilated_conv1]
    for index, shifts in enumerate([(1, 1, 0), (1, 0, 1), (0, 1, 1)]):
        dilated_list.append(dilate_tensor(
            conv2_list[index], axis, shifts, scope+'/dialte_conv2_%s' % index))
    for index, shifts in enumerate([(1, 0, 0), (0, 0, 1), (0, 1, 0)]):
        dilated_list.append(dilate_tensor(
            conv3_list[index], axis, shifts, scope+'/dialte_conv3_%s' % index))
    outputs = tf.add_n(dilated_list, name=scope+'/add')
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs


def voxel_dcl3d(inputs, out_num, kernel_size, scope, action='concat', activation_fn=tf.nn.relu):
    """
    inputs: input tensor
    out_num: output channel number
    kernel_size: convolutional kernel size
    scope: operation scope
    activation_fn: activation function, could be None if needed
    """
    axis, c_axis = (1, 2, 3), 4  # only support format 'NDHWC'
    conv0 = conv3d(inputs, out_num, kernel_size, scope+'/conv0')
    conv1 = conv3d(conv0, out_num, kernel_size, scope+'/conv1')
    combine1 = combine([conv0, conv1], action, c_axis, scope=scope+'/combine1')
    conv2 = conv3d(combine1, 3*out_num, kernel_size, scope+'/conv2')
    conv2_list = tf.split(conv2, 3, c_axis, name=scope+'/split1')
    combine2 = combine([conv0]+conv2_list, action, c_axis, scope=scope+'/combine2')
    conv3 = conv3d(combine2, 3*out_num, kernel_size, scope+'/conv3')
    conv3_list = tf.split(conv3, 3, c_axis, name=scope+'/split1')
    dilated_conv0 = dilate_tensor(
        conv0, axis, (0, 0, 0), scope+'/dialte_conv0')
    dilated_conv1 = dilate_tensor(
        conv1, axis, (1, 1, 1), scope+'/dialte_conv1')
    dilated_list = [dilated_conv0, dilated_conv1]
    for index, shifts in enumerate([(1, 1, 0), (1, 0, 1), (0, 1, 1)]):
        dilated_list.append(dilate_tensor(
            conv2_list[index], axis, shifts, scope+'/dialte_conv2_%s' % index))
    for index, shifts in enumerate([(1, 0, 0), (0, 0, 1), (0, 1, 0)]):
        dilated_list.append(dilate_tensor(
            conv3_list[index], axis, shifts, scope+'/dialte_conv3_%s' % index))
    outputs = tf.add_n(dilated_list, name=scope+'/add')
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs


def combine(tensors, action, axis, scope):
    if action == 'concat':
        return tf.concat(tensors, axis, name=scope)
    else:
        return tf.add_n(tensors, name=scope)

def conv3d(inputs, out_num, kernel_size, scope):
    shape = list(kernel_size) + [inputs.shape[-1].value, out_num]
    weights = tf.get_variable(
        scope+'/conv/weights', shape, initializer=tf.truncated_normal_initializer())
    outputs = tf.nn.conv3d(
        inputs, weights, (1, 1, 1, 1, 1), padding='SAME', name=scope+'/conv')
    return outputs


def get_mask(shape, scope):
    new_shape = (np.prod(shape[:-2]), shape[-2], shape[-1])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32, name=scope+'/mask')


def dilate_tensor(inputs, axes, shifts, scope):
    for index, axis in enumerate(axes):
        eles = tf.unstack(inputs, axis=axis, name=scope+'/unstack%s' % index)
        zeros = tf.zeros_like(
            eles[0], dtype=tf.float32, name=scope+'/zeros%s' % index)
        for ele_index in range(len(eles), 0, -1):
            eles.insert(ele_index-shifts[index], zeros)
        inputs = tf.stack(eles, axis=axis, name=scope+'/stack%s' % index)
    return inputs


def shift_tensor(inputs, axes, row_shift, column_shift, scope):
    if row_shift:
        rows = tf.unstack(inputs, axis=axes[0], name=scope+'/rowsunstack')
        row_zeros = tf.zeros_like(
            rows[0], dtype=tf.float32, name=scope+'/rowzeros')
        rows = rows[row_shift:] + [row_zeros]*row_shift
        inputs = tf.stack(rows, axis=axes[0], name=scope+'/rowsstack')
    if column_shift:
        columns = tf.unstack(
            inputs, axis=axes[1], name=scope+'/columnsunstack')
        columns_zeros = tf.zeros_like(
            columns[0], dtype=tf.float32, name=scope+'/columnzeros')
        columns = columns[column_shift:] + [columns_zeros]*column_shift
        inputs = tf.stack(columns, axis=axes[1], name=scope+'/columnsstack')
    return inputs
