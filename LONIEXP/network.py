import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, H53DDataLoader
from utils.img_utils import imsave
from utils import ops
import h5py
"""
This module build a standard U-NET with VoxelDCL 
for semantic segmentation.
"""
class VoxelDCN(object):
    def __init__(self, sess, conf, types='train'):
        self.sess = sess
        print("start conf")
        self.conf = conf
        print("finish conf")
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
        if types == 'predict' and self.conf.test_step > 0:
            self.reload(self.conf.test_step)

    def def_params(self):
        self.data_format = 'NHWC'
        self.conv_size = (3, 3, 3)
        self.pool_size = (1, 2, 2)
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
        expand_annotations = tf.expand_dims(
            self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(
            expand_annotations, axis=[self.channel_axis],
            name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(
            one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        losses = tf.losses.softmax_cross_entropy(
            one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        self.decoded_predictions = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        #self.dice_accuracy_op, self.sub_dice_list = ops.dice_accuracy(self.decoded_predictions,\
        #                        self.annotations,self.conf.class_num)
        correct_prediction = tf.equal(
            self.annotations, self.decoded_predictions,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        if name == 'valid' and self.conf.data_type=='2D':
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
        down_outputs.append(conv2)
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
        train_reader = H53DDataLoader(
                self.conf.data_dir+self.conf.train_data, self.input_shape)
        valid_reader = H53DDataLoader(
                self.conf.data_dir+self.conf.valid_data, self.input_shape)
        for epoch_num in range(self.conf.max_step):
            if epoch_num % self.conf.test_interval == 0:   
                inputs, annotations = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, summary, accuracy = self.sess.run(
                    [self.loss_op, self.valid_summary,self.accuracy_op], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
                print('----valid loss', loss)
                print('----valid accuracy', accuracy)
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
                loss, summary, _, accuracy = self.sess.run(
                    [self.loss_op, self.train_summary, self.train_op, self.accuracy_op
                    ], feed_dict=feed_dict)
                print('----train loss', loss)
                print('----train accuracy', accuracy)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            if epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num+self.conf.reload_step)

    def predict_func(self,predict_generator):
        s_predictions = []
        s_labels = []
        print("start Depth------>")
        for s_start in range(0,self.conf.predict_batch-self.conf.predict_batch%self.conf.d_gap+1,self.conf.d_gap):
            s_start = min(s_start, self.conf.predict_batch - self.conf.depth)
            h_predictions = []
            h_labels = []
            print("start Height------------>")
            for h_start in range(0,self.conf.data_height-self.conf.data_height%self.conf.w_gap+1,self.conf.h_gap):
                h_start = min(h_start, self.conf.data_height - self.conf.height)
                w_predictions = []
                w_labels = []
                print("start Width------------------>")
                for w_start in range(0,self.conf.data_width-self.conf.data_width%self.conf.h_gap+1,self.conf.w_gap):
                    w_start = min(w_start,self.conf.data_width-self.conf.width)
                    inputs, annotations = next(predict_generator)
                    feed_dict = {self.inputs: inputs, self.annotations: annotations}
                    prediction, loss, accuracy = self.sess.run(
                        [self.predictions, self.loss_op, self.accuracy_op],
                        feed_dict=feed_dict)
                    prediction = tf.unstack(prediction, axis=3)
                    if w_start == 0:
                        w_predictions = w_predictions + prediction
                        w_labels = annotations
                    else:
                        w_overlap = len(w_predictions) - w_start
                        for i in range(w_overlap):
                            w_predictions[i-w_overlap] = w_predictions[i-w_overlap]*(1-i*1.0/w_overlap)\
                                + prediction[i]*(i*1.0/w_overlap)
                        w_predictions = w_predictions+prediction[w_overlap:]
                        w_labels = tf.concat([w_labels,annotations[:,:,:,w_overlap:]], axis=3)
                w_predictions = tf.stack(w_predictions, axis=3)
                w_predictions = tf.unstack(w_predictions, axis=2)
                if h_start == 0:
                    h_predictions = h_predictions + w_predictions
                    h_labels = w_labels
                else:
                    h_overlap = len(h_predictions) - h_start
                    for i in range(h_overlap):
                        h_predictions[i-h_overlap] = h_predictions[i-h_overlap]*(1-i*1.0/h_overlap)\
                            + w_predictions[i]*(i*1.0/h_overlap)
                    h_predictions = h_predictions+w_predictions[h_overlap:]
                    h_labels = tf.concat([h_labels,w_labels[:,:,h_overlap:,:]], axis=2)
            h_predictions = tf.stack(h_predictions,axis = 2)
            h_predictions = tf.unstack(h_predictions,axis = 1)
            if s_start == 0:
                s_predictions = s_predictions + h_predictions
                s_labels = h_labels
            else:
                s_overlap = len(s_predictions) - s_start
                for i in range(s_overlap):
                    s_predictions[i-s_overlap] = s_predictions[i-s_overlap]*(1-i*1.0/s_overlap)\
                        + h_predictions[i]*(i*1.0/s_overlap)
                s_predictions = s_predictions+h_predictions[s_overlap:]
                s_labels = tf.concat([s_labels,h_labels[:,s_overlap:,:,:]],axis = 1)
        s_predictions = tf.stack(s_predictions, axis=1)
        s_labels = tf.cast(s_labels,'uint8')
        return  s_predictions, s_labels    

    def predict(self,test_step,data_index,sub_batch_index, test_type):
        print('---->predicting ', self.conf.test_step)
        #if test_type == 'predict' and self.conf.test_step > 0:
	#    print("start reload")
	#    self.reload(self.conf.test_step)
	#elif test_type == 'valid':
	#    self.reload(test_step)
        #else:
        #    print("please set a reasonable test_step")
        #    return  
        if test_type == 'valid':
            test_reader = H53DDataLoader(
                self.conf.data_dir+self.conf.valid_data, self.input_shape,is_train=False)           
        elif test_type == 'predict':
            print("start get predict data")
            test_reader = H53DDataLoader(
                self.conf.data_dir+self.conf.test_data, self.input_shape,is_train=False)   
            print("finish get data")
        else:
            print("invalid type")
            return
        predict_generator = test_reader.generate_data(data_index,sub_batch_index,[self.conf.depth,self.conf.height,self.conf.width],[self.conf.d_gap,self.conf.w_gap,self.conf.h_gap])
        s_predictions,s_labels = self.predict_func(predict_generator)
        print("finish concate ---------+++++")
	# process label and data 
        # process label
        expand_annotations = tf.expand_dims(
            s_labels, -1, name='s_labels/expand_dims')
        one_hot_annotations = tf.squeeze(
            expand_annotations, axis=[self.channel_axis],
            name='s_labels/squeeze')
        one_hot_annotations = tf.one_hot(
            one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='s_labels/one_hot')
        # process data
        decoded_predictions = tf.argmax(
            s_predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            tf.cast(s_labels,tf.int64), decoded_predictions,
            name='accuracy/predict_correct_pred')
        #accuracy
        accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        #loss
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot_annotations, s_predictions))
        #dice ratio
        DiceRatio, sublist = ops.dice_accuracy(decoded_predictions, s_labels, self.conf.class_num)
        print("session image labels-------vvvvvvv")
        accuracy, Loss, diceRatio, Sublist = self.sess.run([accuracy_op, loss,DiceRatio,sublist])
        del DiceRatio, sublist
        return accuracy, Loss, diceRatio, Sublist

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        print("reload:",step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

if __name__ == '__main__':
    a = [1,3,4]

