
import numpy as np
import tensorflow as tf
import os
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras import backend as K
# from keras import Input
# from keras.layers import Dense, Lambda
# from keras.layers import Add, Reshape
# from keras.models import Model
# import tensorflow.keras.backend as K
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
            n_output_survival=2,
            n_length=64,
            channel=3,
            learning_rate=0.00001,
            batch_size=5,
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
#=====================================
    def MultiHeadsAttModel(self, l=1 *7 * 7, d=32*3, dv=8*3, dout=32, nv=8):
        v1 = tf.keras.Input(shape=(l, d))
        q1 = tf.keras.Input(shape=(l, d))
        k1 = tf.keras.Input(shape=(l, d))

        v2 = tf.keras.layers.Dense(dv * nv, activation="relu")(v1)
        q2 = tf.keras.layers.Dense(dv * nv, activation="relu")(q1)
        k2 = tf.keras.layers.Dense(dv * nv, activation="relu")(k1)

        v = tf.keras.layers.Reshape([l, nv, dv])(v2)
        q = tf.keras.layers.Reshape([l, nv, dv])(q2)
        k = tf.keras.layers.Reshape([l, nv, dv])(k2)


        att = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0],x[1] ,axes=[1,1]) / np.sqrt(dv),output_shape=(l, nv, nv))([q,k])# l, nv, nv
        # att = tf.einsum('baik,baij->bakj', q, k) / np.sqrt(dv)
        # att = tf.einsum('', q, k)

        att1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.softmax(x), output_shape=(l, nv, nv))(att)

        out = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1],axes=[3,3]),  output_shape=(l, nv, dv))([att1, v])
        # out = tf.einsum('bajk,baik->baji', att1, v)

        out1 = tf.keras.layers.Reshape([l, d])(out)

        out2 = tf.keras.layers.Add()([out1, q1])

        # out3 = tf.keras.layers.Dense(dout, activation="relu")(out2)
        out3 = tf.layers.dense(out2, dout, activation=tf.nn.relu)

        inputs2 = [q1, k1, v1]
        outputs2 = out3
        return tf.keras.Model(inputs=[q1, k1, v1], outputs=[out3])
# #=======================================================
    def _build_net(self,input,is_training):


        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv3d(input,
                                     filters=4,
                                     kernel_size=(5,5,5),
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     # bias_initializer=Constant(value=-0.9),
                                     dilation_rate=1,
                                     )
            bn1 = tf.layers.batch_normalization(conv1, axis=3, training=is_training, renorm=False)
            bn11 = tf.nn.leaky_relu(bn1)
            drop1 = tf.nn.dropout(bn11, 0.5)
            # pool1 = tf.layers.max_pooling3d(inputs=drop1, pool_size=(1, 3, 3), strides=(1, 2, 2))
            # drop11 = tf.nn.dropout(pool1, 0.5)

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv3d(drop1,
                                     filters=8,
                                     kernel_size=(5,5,5),
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn2= tf.layers.batch_normalization(conv2, axis=3, training=is_training, renorm=False)
            bn22 = tf.nn.leaky_relu(bn2)
            # drop2 = tf.nn.dropout(bn22, 0.6)
            pool2 = tf.layers.max_pooling3d(inputs=bn22, pool_size=(2, 3, 3), strides=(1, 2, 2))
            drop22 = tf.nn.dropout(pool2, 0.5)

        with tf.variable_scope('conv3'):
            conv3 = tf.layers.conv3d(drop22,
                                     filters=16,
                                     kernel_size=(4,5,5),
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn3 = tf.layers.batch_normalization(conv3, axis=3, training=is_training, renorm=False)
            bn33 = tf.nn.leaky_relu(bn3)
            drop3 = tf.nn.dropout(bn33, 0.5)
            # pool3 = tf.layers.max_pooling3d(inputs=bn33, pool_size=(2, 3, 3), strides=(1, 2, 2))
            # drop33 = tf.nn.dropout(pool3, 0.5)

        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv3d(drop3,
                                     filters=24,
                                     kernel_size=(3,5,5),
                                     strides = 1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn4 = tf.layers.batch_normalization(conv4, axis=3, training=is_training, renorm=False)
            bn44 = tf.nn.leaky_relu(bn4)
            drop4 = tf.nn.dropout(bn44, 0.5)
            # pool4 = tf.layers.max_pooling3d(inputs=bn44, pool_size=(3, 3, 3), strides=(1, 2, 2))
            # drop44 = tf.nn.dropout(pool4, 0.5)

        with tf.variable_scope('conv5'):
            conv5 = tf.layers.conv3d(drop4,
                                     filters=24,
                                     kernel_size=(3,5,5),
                                     strides = 1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn5 = tf.layers.batch_normalization(conv5, axis=3, training=is_training, renorm=False)
            bn55 = tf.nn.leaky_relu(bn5)
            # drop5 = tf.nn.dropout(bn55, 0.5)
            pool5 = tf.layers.max_pooling3d(inputs=bn55, pool_size=(3, 3, 3), strides=(1, 2, 2))
            drop55 = tf.nn.dropout(pool5, 0.5)

        with tf.variable_scope('conv6'):
            conv6 = tf.layers.conv3d(drop55,
                                     filters=32,
                                     kernel_size=(3, 5, 5),
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn6 = tf.layers.batch_normalization(conv6, axis=3, training=is_training, renorm=False)
            bn66 = tf.nn.leaky_relu(bn6)
            # drop6 = tf.nn.dropout(bn66, 0.5)
            pool6 = tf.layers.max_pooling3d(inputs=bn66, pool_size=(2, 3, 3), strides=(1, 2, 2))
            drop66 = tf.nn.dropout(pool6, 0.5)

        with tf.variable_scope('conv7'):
            conv7 = tf.layers.conv3d(drop66,
                                     filters=48,
                                     kernel_size=(1,7,7),
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     # bias_initializer=Constant(value=-0.9),
                                     dilation_rate=1,
                                     )
            bn7= tf.layers.batch_normalization(conv7, training=is_training, renorm=False)
            bn77 = tf.nn.leaky_relu(bn7)
            drop7 = tf.nn.dropout(bn77, 0.5)
            # pool7 = tf.layers.max_pooling3d(inputs=bn77, pool_size=(1, 2, 2), strides=(1, 2, 2))
            # drop77 = tf.nn.dropout(pool7, 0.5)


#=============================================================
            # print("x out conv shape", drop5.shape)
        if True:
            x = tf.keras.layers.Reshape([1 * 7 * 7, 48])(drop7)
            # xx = tf.reshape(drop6, [-1, 1 * 7 * 7, 64*3 ])
            att = self.MultiHeadsAttModel(l= 1 * 7 * 7, d=48 , dv=4*3, dout=24, nv=4)
            x = att([x, x, x])
            x = tf.keras.layers.Reshape([1, 7, 7, 24])(x)
            # x = tf.reshape(x, [1, 7, 7, 32])
            # x = NormL()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.layers.batch_normalization(x, training=is_training, renorm=False)
#============================================================

        with tf.variable_scope('fc1'):
            flat_input = tf.layers.flatten(x)    #Global Average Pooling
            # fc1 = tf.layers.dense(flat_input, 1024 ,activation=tf.nn.leaky_relu)
            # drop1 = tf.nn.dropout(fc1, 0.5)

            fc2 = tf.layers.dense(flat_input, 512, activation=tf.nn.leaky_relu)
            drop2 = tf.nn.dropout(fc2, 0.5)

            fc3 = tf.layers.dense(drop2, 256, activation=tf.nn.leaky_relu)
            drop3 = tf.nn.dropout(fc3, 0.5)

            fc4 = tf.layers.dense(drop3, 128, activation=tf.nn.leaky_relu)
            drop4 = tf.nn.dropout(fc4, 0.5)

            fc5 = tf.layers.dense(drop4, 64, activation=tf.nn.leaky_relu)
            drop5 = tf.nn.dropout(fc5, 0.5)

        with tf.variable_scope('fcg'):  # out  grad
            fcg = tf.layers.dense(drop5, 2, activation=tf.nn.leaky_relu)
            out_grad = tf.nn.softmax(fcg)

        with tf.variable_scope('fcc'):  # out survival
            fcs = tf.layers.dense(drop5, 2, activation=tf.nn.leaky_relu)
            out_survival = tf.nn.softmax(fcs)

        return out_grad, out_survival





