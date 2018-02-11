import os
import time
import argparse
import tensorflow as tf
from network import VoxelDCN

"""
This file provides configuration to build U-NET with VoxelDCL for 3D semantic segmentation.

"""
dir = '../sampledata/ADNI/'

def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 100001,'# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 1000, '# of interval to save a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    # data
    flags.DEFINE_string('data_dir', dir, 'Name of data directory')
    flags.DEFINE_string('train_data', 'train_0.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'test_0.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'test_0.h5', 'Testing data')
    flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
    flags.DEFINE_integer('batch', 4, 'batch size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('depth', 96, 'depth size')
    flags.DEFINE_integer('height', 124, 'height size')
    flags.DEFINE_integer('width', 96, 'width size')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 2500, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_depth', 3, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 2, 'output class number')
    flags.DEFINE_integer('start_channel_num', 16,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv',
        'Use which conv op in decoder: currently only convolution operation can be used')
    #dilate-deconv
    flags.DEFINE_string(
        'deconv_name', 'voxel_dcl',
        'Use which deconv op in decoder: deconv, voxel_dcl, ivoxel_dcl')
    flags.DEFINE_string(
    'action', 'concat',
    'Use how to combine feature maps in voxel_dcl and ivoxel_dcl: concat or add')
    # Dense Transformer Networks
    flags.DEFINE_boolean('add_dtn', False,
        'add Dense Transformer Networks or not')
    flags.DEFINE_integer('dtn_location', 1,'The Dense Transformer Networks location')
    flags.DEFINE_string('control_points_ratio', 2,
        'Setup the ratio of control_points comparing with the Dense transformer networks input size')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='option: train, test, or predict')
    args = parser.parse_args()
    if args.option not in ['train', 'test', 'predict']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")
    else:
        model = VoxelDCN(tf.Session(), configure())
        getattr(model, args.option)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tf.app.run()
