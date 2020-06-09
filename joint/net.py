
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
            conv = tf.layers.conv3d(input,
                                     filters=8,
                                     kernel_size=3,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            pool1 = tf.layers.max_pooling3d(inputs=bn, pool_size=2, strides=2)
        with tf.variable_scope('conv2'):
            conv = tf.layers.conv3d(pool1,
                                     filters=16,
                                     kernel_size=3,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
            bn1 = tf.nn.leaky_relu(bn)
            # pool1 = tf.layers.max_pooling3d(inputs=bn1, pool_size=2, strides=2)

        with tf.variable_scope('conv3'):
            conv = tf.layers.conv3d(bn1,
                                     filters=16,
                                     kernel_size=3,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            pool2 = tf.layers.max_pooling3d(inputs=bn, pool_size=2, strides=2)
        with tf.variable_scope('conv4'):
            conv = tf.layers.conv3d(pool2,
                                     filters=32,
                                     kernel_size=3,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
            bn3 = tf.nn.leaky_relu(bn)
            # pool3 = tf.layers.max_pooling3d(inputs=bn3, pool_size=2, strides=2)


        with tf.variable_scope('conv5'):
            conv = tf.layers.conv3d(bn3,
                                     filters=64,
                                     kernel_size=3,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
            bn5 = tf.nn.leaky_relu(bn)

        with tf.variable_scope('fc1'):
            flat_inputs = tf.layers.flatten(bn5)
            nn = tf.layers.dense(flat_inputs, 512 ,activation=tf.nn.leaky_relu)

        with tf.variable_scope('fc2'):   # for grad
            fc2 = tf.layers.dense(nn, 64, activation=tf.nn.leaky_relu)

        with tf.variable_scope('fc3'):  # for survival
            fc3 = tf.layers.dense(nn, 64, activation=tf.nn.leaky_relu)

        with tf.variable_scope('fc22'):  # out  grad
            fc22 = tf.layers.dense(fc2, 2, activation=tf.nn.leaky_relu)
            out_grad = tf.nn.softmax(fc22)

        with tf.variable_scope('fc33'):  # out survival
            fc33 = tf.layers.dense(fc3, 3, activation=tf.nn.leaky_relu)
            out_survival = tf.nn.softmax(fc33)

            return out_grad, out_survival

            # return tf.nn.softmax(nn)


