
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

np.random.seed(1)
tf.set_random_seed(1)
from shutil import copyfile

import shutil

# Deep Q Network off-policy
class Network(object):
    def __init__(
            self,logno,
            n_output_grad=2,
            n_output_survival=3,
            n_length=64,
            channel=3,
            learning_rate=0.001,
            batch_size=128,
            output_graph=False,
            use_ckpt = True

    ):
        learn_step_counter = 0,
        global_step = tf.Variable(tf.constant(1)),
        global_counter = 1 # equal to self.global_step




        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        self.dir_path = '/exports/lkeb-hpc/gkarami/Code/Log/'+str(logno)
        os.mkdir(self.dir_path)
        os.mkdir(self.dir_path+'/code/')
        os.mkdir(self.dir_path+'/code/joint/')

        copyfile('./net.py', self.dir_path + '/code/joint/net.py')
        copyfile('./real_time.py', self.dir_path + '/code/joint/real_time.py')
        copyfile('./test_model.py', self.dir_path + '/code/joint/test_model.py')
        copyfile('./Utils.py', self.dir_path + '/code/joint/Utils.py')
        copyfile('./jointly.py', self.dir_path + '/code/joint/jointly.py')




        if use_ckpt:
            self.restore_parameters()
        else:
            self.sess.run(tf.global_variables_initializer()) # train step

        self.cost_his = []


    def _build_net(self,input,is_training):


        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv3d(input,
                                     filters=8,
                                     kernel_size=7,
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     # bias_initializer=Constant(value=-0.9),
                                     dilation_rate=1,
                                     )
            bn1 = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn1 = tf.nn.leaky_relu(bn1)
            drop1 = tf.nn.dropout(bn1, 0.5)
            # pool1 = tf.layers.max_pooling3d(inputs=drop1, pool_size=(1, 3, 3), strides=(1, 2, 2))


        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv3d(drop1,
                                     filters=16,
                                     kernel_size=7,
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn2= tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            bn2 = tf.nn.leaky_relu(bn2)
            drop2 = tf.nn.dropout(bn2, 0.5)
            pool2 = tf.layers.max_pooling3d(inputs=drop2, pool_size=(1, 3, 3), strides=(1, 2, 2))
            drop2 = tf.nn.dropout(pool2, 0.5)

        with tf.variable_scope('conv3'):
            conv3 = tf.layers.conv3d(drop2,
                                     filters=24,
                                     kernel_size=7,
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn3 = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            bn3 = tf.nn.leaky_relu(bn3)
            drop3 = tf.nn.dropout(bn3, 0.5)
            pool3 = tf.layers.max_pooling3d(inputs=drop3, pool_size=(3, 3, 3), strides=(1, 2, 2))
            drop3 = tf.nn.dropout(pool3, 0.5)

        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv3d(drop3,
                                     filters=32,
                                     kernel_size=3,
                                     strides = 1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn4 = tf.layers.batch_normalization(conv4, training=is_training, renorm=False)
            bn4 = tf.nn.leaky_relu(bn4)
            drop4 = tf.nn.dropout(bn4, 0.5)
            pool4 = tf.layers.max_pooling3d(inputs=drop4, pool_size=(3, 3, 3), strides=(1, 2, 2))
            drop4 = tf.nn.dropout(pool4, 0.5)

        with tf.variable_scope('conv5'):
            conv5 = tf.layers.conv3d(drop4,
                                     filters=48,
                                     kernel_size=7,
                                     strides = 1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn5 = tf.layers.batch_normalization(conv5, training=is_training, renorm=False)
            bn5 = tf.nn.leaky_relu(bn5)
            drop5 = tf.nn.dropout(bn5, 0.5)
            # pool5 = tf.layers.max_pooling3d(inputs=bn5, pool_size=2, strides=(1, 2, 2))

        with tf.variable_scope('fc1'):
            flat_input = tf.layers.flatten(drop5)
            fc1 = tf.layers.dense(flat_input, 256 ,activation=tf.nn.leaky_relu)
            drop1 = tf.nn.dropout(fc1, 0.5)

        with tf.variable_scope('fc2'):
            fc2 = tf.layers.dense(drop1, 64, activation=tf.nn.leaky_relu)
            drop2 = tf.nn.dropout(fc2, 0.5)

        with tf.variable_scope('fc22'):  # out  grad
            fc22 = tf.layers.dense(drop2, 2, activation=tf.nn.leaky_relu)
            out_grad = tf.nn.softmax(fc22)
            # out_survival = tf.nn.softmax(fc22)

        return out_grad



