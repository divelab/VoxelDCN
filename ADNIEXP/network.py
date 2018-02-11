import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, H53DDataLoader
from utils.img_utils import imsave
from utils import ops


"""
This module builds a standard U-NET with VoxelDCL for 3D 
semantic segmentation.
"""


class VoxelDCN(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        self.data_format = 'NHWC'
        self.conv_size = (3, 3, 3)
        self.pool_size = (2, 2, 2)
        self.axis, self.channel_axis = (1, 2, 3), 4
        self.input_shape = [
            self.conf.batch, self.conf.depth, self.conf.height,
            self.conf.width, self.conf.channel]
        self.output_shape = [
            self.conf.batch, self.conf.depth, self.conf.height,
            self.conf.width]

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(
            tf.int64, self.output_shape, name='annotations')
        self.predictions = self.inference(self.inputs)
        self.cal_loss()

    def cal_loss(self):
        one_hot_annotations = tf.one_hot(
            self.annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        losses = tf.losses.softmax_cross_entropy(
            one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        self.decoded_predictions = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.annotations, self.decoded_predictions,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        weights = tf.cast(
            tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
            tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(
            self.annotations, self.decoded_predictions, self.conf.class_num,
            weights, name='m_iou/m_ious')

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        if name == 'valid' and self.conf.data_type == '2D':
            summarys.append(
                tf.summary.image(name+'/input', self.inputs, max_outputs=100))
            summarys.append(
                tf.summary.image(
                    name+'/annotation',
                    tf.cast(tf.expand_dims(self.annotations, -1),
                            tf.float32), max_outputs=100))
            summarys.append(
                tf.summary.image(
                    name+'/prediction',
                    tf.cast(tf.expand_dims(self.decoded_predictions, -1),
                            tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            outputs = self.build_down_block(outputs, name, down_outputs, first=is_first)
        print("down ",layer_index," shape ", outputs.get_shape())
        outputs = self.build_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.build_up_block(outputs, down_inputs, name,final=is_final)
        print("up ",layer_index," shape ",outputs.get_shape())
        return outputs

    def build_down_block(self, inputs, name, down_outputs, first=False):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = ops.conv(inputs, out_num, self.conv_size,
                         name+'/conv1', self.conf.data_type)
        conv2 = ops.conv(conv1, out_num, self.conv_size,
                         name+'/conv2', self.conf.data_type)
        down_outputs.append(conv1)
        pool = ops.pool(conv2, self.pool_size, name +
                        '/pool', self.conf.data_type)
        return pool

    def build_bottom_block(self, inputs, name):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = ops.conv(
            inputs, 2*out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv2 = ops.conv(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        return conv2

    def build_up_block(self, inputs, down_inputs, name, final=False):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, out_num, self.conv_size, name+'/conv1',
            self.conf.data_type, action=self.conf.action)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.conv_func()(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        out_num = self.conf.class_num if final else out_num/2
        conv3 = ops.conv(
            conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        train_reader = H5DataLoader(
            self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(
            self.conf.data_dir+self.conf.valid_data)
        iteration = train_reader.iter
        pre_iter = iteration
        epoch_num = 0
        while iteration < self.conf.max_step:
            if pre_iter != iteration:
                pre_iter = iteration
                inputs, annotations = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, summary = self.sess.run(
                    [self.loss_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, iteration)
                self.save(iteration)
                print('----testing loss', loss)
            elif epoch_num % self.conf.summary_interval == 0:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            else:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                print('----training loss', loss)
            iteration = train_reader.iter
            epoch_num += 1

    def test(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        test_reader = H5DataLoader(
            self.conf.data_dir+self.conf.test_data, True)
        predictions = []
        labels = []
        while test_reader.iter <1:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
            res,acc = self.sess.run([self.decoded_predictions, self.accuracy_op],feed_dict = feed_dict)
            print(acc)
            #res = np.concatenate(res,axis=0)
            predictions.append(res)
            labels.append(annotations)
        predictions = np.concatenate(predictions,axis=0)
        labels = np.concatenate(labels,axis=0)
        print('---',predictions.shape)
        print(labels.shape)
        np.savez('temp',predictions,labels)
        print(predictions.shape)
        print(ops.dice_ratio(predictions[0],labels[0]))
        print(ops.dice_ratio(predictions[1],labels[1]))

    def predict(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        test_reader = H5DataLoader(
            self.conf.data_dir+self.conf.test_data, False)
        predictions = []
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            predictions.append(self.sess.run(
                self.decoded_predictions, feed_dict=feed_dict))
        print('----->saving predictions')
        for index, prediction in enumerate(predictions):
            for i in range(prediction.shape[0]):
                imsave(prediction[i], self.conf.sampledir +
                       str(index*prediction.shape[0]+i)+'.png')

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
