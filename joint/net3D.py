
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
            n_volume=5,
            channel=3,
            learning_rate=0.001,
            batch_size=128,
            output_graph=False,
            use_ckpt = True,
            DTYPE=tf.float32

    ):


        self.n_length = n_length # width or height of input matrix
        self.n_volume= n_volume,
        self.lr = learning_rate
        self.batch_size = batch_size
        self.channel = channel # num of channel
        self.learn_step_counter = 0
        self.global_step = tf.Variable(tf.constant(1))
        self.global_counter = 1 # equal to self.global_step
        self.n_output_survival = n_output_survival
        self.n_output_grad = n_output_grad
        self._build_net()
        self.keep_prob = tf.placeholder(tf.float32)
        self.DTYPE=DTYPE

        # e_params = tf.get_collection('eval_net_params')

        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # self.replace_eval_op = [tf.assign(e, t) for e, t in zip(e_params,t_params)]
        # assign e to t

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        # self.dir_path = '/home/hengtong/project/protein/ProteinHP_DQN/heng_model/save_weight/'
        self.dir_path = '/exports/lkeb-hpc/gkarami/Code/Log/'+str(logno)#os.path.dirname(os.path.realpath(__file__))
        os.mkdir(self.dir_path)
        os.mkdir(self.dir_path+'/code/')
        os.mkdir(self.dir_path+'/code/joint/')

        copyfile('./net.py', self.dir_path + '/code/joint/net.py')
        copyfile('./real_time.py', self.dir_path + '/code/joint/real_time.py')
        copyfile('./test_model.py', self.dir_path + '/code/joint/test_model.py')
        copyfile('./Utils.py', self.dir_path + '/code/joint/Utils.py')
        copyfile('./jointly.py', self.dir_path + '/code/joint/jointly.py')


        self.train_writer = tf.summary.FileWriter((self.dir_path)+'/train/' , graph=tf.get_default_graph())
        self.validation_writer = tf.summary.FileWriter((self.dir_path)+'/validation/' , graph=self.sess.graph)

        self.merged = tf.summary.merge_all()
        if output_graph:
            # tensorboard --logdir=logs
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        if use_ckpt:
            self.restore_parameters()
        else:
            self.sess.run(tf.global_variables_initializer()) # train step

        self.cost_his = []

    def get_nb_params_shape(self, shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    def count_number_trainable_params(self):
        '''
        Counts the number of trainable variables.
        '''
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params


    # def conv2d(self,x, W, stride,pad):
    #     # stride [1, x_movement, y_movement, 1]
    #     # Must have strides[0] = strides[3] = 1
    #     return tf.nn.conv2d(x, W, strides = stride, padding=pad)
    #
    # def max_pool(self,x,k,stride,pad):
    #     # stride [1, x_movement, y_movement, 1]
    #     return tf.nn.max_pool(x, ksize=k, strides=stride, padding=pad)

    def BN_fc(self,x,dim):
        # x is input size,dim is batch size
        mean_value,var_value = tf.nn.moments(x,[0])
        scales = tf.Variable(tf.ones([dim]))
        betas = tf.Variable(tf.zeros([dim]))
        epsilon = 1e-3
        return tf.nn.batch_normalization(x,mean_value,var_value,scales,betas,epsilon)



    def _build_net(self):
        # ------------------ build grad_Net ------------------
        self.xs = tf.placeholder(tf.float32,  shape=(5, self.n_length, self.n_length, self.channel), name='input')  # input
        self.xs2 = tf.placeholder(tf.float32, shape=(None, 5, 64, 64, 3), name='input')  # input

        # self.s = tf.reshape(self.xs, [-1, self.n_features, self.n_features, self.channel])
        self.keep_prob = tf.placeholder(tf.float32)
        self.labels_grad = tf.placeholder(tf.float32,[None,self.n_output_grad],name='grad_labels')
        self.labels_survival = tf.placeholder(tf.float32, [None, self.n_output_survival], name='survival_labels')

        self.loss_ts = tf.placeholder(tf.float32)


        with tf.variable_scope('multi-para'):
            self.p = tf.Variable(0.5, name='p')
            self.q = tf.Variable(0.5, name='q')
            tf.summary.scalar('multi-para' + 'p', self.p)
            tf.summary.scalar('multi-para' + 'q', self.q)
        tf.summary.scalar("loss_ts", self.loss_ts)

        show_img = self.xs[:, :, :, 0, np.newaxis]
        tf.summary.image('00: input_t1', show_img, 3)

        show_img = self.xs[:, :, :, 1, np.newaxis]
        tf.summary.image('01: input_md', show_img, 3)

        show_img = self.xs[:, :, :, 2, np.newaxis]
        tf.summary.image('02: input_cbv', show_img, 3)


        with tf.variable_scope('joint_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['grad_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.truncated_normal_initializer(0., 0.1), tf.constant_initializer(0.1)  # config of layers

##################################################################################################################
            def _weight_variable(name, shape):
                return tf.get_variable(name, shape, tf.truncated_normal_initializer(stddev=0.1))

            def _bias_variable(name, shape):
                return tf.get_variable(name, shape, tf.constant_initializer(0.1, dtype=self.DTYPE))

            # first layer. collections is used later when assign to target net
            prev_layer = self.xs2
            in_filters = self.channel
            with tf.variable_scope('conv1') as scope:
                out_filters = 16
                kernel = tf.get_variable('weights', [5, 5, 5, 3, out_filters], tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', [1, out_filters])
                bias = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(bias, name=scope.name)

                input = conv1
                in_filters = out_filters
                pool1 = tf.nn.max_pool3d(input, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')  ############## 32*32*
                norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
                input = norm1

            with tf.variable_scope('conv2') as scope:
                out_filters = 32
                kernel2 = _weight_variable('weights2', [5, 5, 5, in_filters, out_filters])
                conv = tf.nn.conv3d(prev_layer, kernel2, [1, 1, 1, 1, 1], padding='SAME')
                biases2 = _bias_variable('biases2', [out_filters])
                bias = tf.nn.bias_add(conv, biases2)
                conv2 = tf.nn.relu(bias, name=scope.name)

                prev_layer = conv2
                in_filters = out_filters
                # normalize prev_layer here
                prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')  #### 16*16*

            with tf.variable_scope('conv3_1') as scope:
                out_filters = 64
                kernel31 = _weight_variable('weights31', [5, 5, 5, in_filters, out_filters])
                conv = tf.nn.conv3d(prev_layer, kernel31, [1, 1, 1, 1, 1], padding='SAME')
                biases31 = _bias_variable('biases31', [out_filters])
                bias = tf.nn.bias_add(conv, biases31)
                prev_layer = tf.nn.relu(bias, name=scope.name)
                in_filters = out_filters

            with tf.variable_scope('conv3_2') as scope:
                out_filters = 64
                kernel32 = _weight_variable('weights32', [5, 5, 5, in_filters, out_filters])
                conv = tf.nn.conv3d(prev_layer, kernel32, [1, 1, 1, 1, 1], padding='SAME')
                biases32 = _bias_variable('biases32', [out_filters])
                bias = tf.nn.bias_add(conv, biases32)
                prev_layer = tf.nn.relu(bias, name=scope.name)
                in_filters = out_filters

            with tf.variable_scope('conv3_3') as scope:
                out_filters = 32
                kernel33 = _weight_variable('weights33', [5, 5, 5, in_filters, out_filters])
                conv = tf.nn.conv3d(prev_layer, kernel33, [1, 1, 1, 1, 1], padding='SAME')
                biases33 = _bias_variable('biases33', [out_filters])
                bias = tf.nn.bias_add(conv, biases33)
                prev_layer = tf.nn.relu(bias, name=scope.name)
                in_filters = out_filters

                # normalize prev_layer here
                prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')  #### 8*8**

            # fully connected layer 1
            with tf.variable_scope('fcl1'):
                w1_fu = tf.get_variable('w1_fu',[1*8*8*32,1024],initializer=w_initializer, collections=c_names)
                b1_fu = tf.get_variable('b1_fu',[1,1024],initializer=b_initializer, collections=c_names)
                h_pool3_flat = tf.reshape(prev_layer, [-1,1*8*8*32])
                bn_in_fc1 = tf.matmul(h_pool3_flat, w1_fu) + b1_fu
                # bn_out_fc1 = self.BN_fc(bn_in_fc1,512)
                h_fc1 = tf.nn.relu(bn_in_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)#################################
                tf.summary.histogram('fcl1' + '/weight', w1_fu)
                tf.summary.histogram('fcl1' + '/bias', b1_fu)

            # fully connected layer 2,grad
            with tf.variable_scope('fcl2_grad'):
                w2_fc_grad = tf.get_variable('w2_fc_grad', [1024,32], initializer=w_initializer, collections=c_names)
                b2_fc_grad = tf.get_variable('b2_fc_grad', [1,32], initializer=b_initializer, collections=c_names)
                bn_fc2_grad = tf.matmul(h_fc1_drop, w2_fc_grad) + b2_fc_grad
                # self.q_eval = self.BN_fc(bn_in_fc2, self.n_actions)
                h_fc2_grad = tf.nn.relu(bn_fc2_grad)
                h_fc2_grad_drop = tf.nn.dropout(h_fc2_grad, self.keep_prob)
                tf.summary.histogram('fcl2_grad' + '/weight', w2_fc_grad)
                tf.summary.histogram('fcl2_grad' + '/bias', b2_fc_grad)

            # fully connected layer 2,survival
            with tf.variable_scope('fcl2_survival'):
                w2_fc_survival = tf.get_variable('w2_fc_survival', [1024, 32], initializer=w_initializer, collections=c_names)
                b2_fc_survival = tf.get_variable('b2_fc_survival', [1, 32], initializer=b_initializer, collections=c_names)
                bn_fc2_survival = tf.matmul(h_fc1_drop, w2_fc_survival) + b2_fc_survival
                h_fc2_survival = tf.nn.relu(bn_fc2_survival)

                h_fc2_survival_drop = tf.nn.dropout(h_fc2_survival, self.keep_prob)
                tf.summary.histogram('fcl2_survival' + '/weight', w2_fc_survival)
                tf.summary.histogram('fcl2_survival' + '/bias', b2_fc_survival)

            # output layer,grad
            with tf.variable_scope('out_grad'):
                w3_fc_grad = tf.get_variable('w3_fc_grad', [32, self.n_output_grad], initializer=w_initializer, collections=c_names)
                b3_fc_grad = tf.get_variable('b3_fc_grad', [1, self.n_output_grad], initializer=b_initializer, collections=c_names)
                bn_fc3_grad = tf.matmul(h_fc2_grad_drop, w3_fc_grad) + b3_fc_grad
                # self.q_eval = self.BN_fc(bn_in_fc2, self.n_actions)
                self.out_grad = tf.multiply(self.q, tf.nn.softmax(bn_fc3_grad))
                # self.out_grad = tf.nn.softmax(bn_fc3_grad)
                tf.summary.histogram('out_grad' + '/weight', w3_fc_grad)
                tf.summary.histogram('out_grad' + '/bias', b3_fc_grad)

            # output layer,survival
            with tf.variable_scope('out_survival'):
                w3_fc_survival = tf.get_variable('w3_fc_survival', [32, self.n_output_survival], initializer=w_initializer, collections=c_names)
                b3_fc_survival = tf.get_variable('b3_fc_survival', [1, self.n_output_survival], initializer=b_initializer, collections=c_names)
                bn_fc3_survival = tf.matmul(h_fc2_survival_drop, w3_fc_survival) + b3_fc_survival
                self.out_survival = tf.multiply(self.p, tf.nn.softmax(bn_fc3_survival))
                # self.out_survival =  tf.nn.softmax(bn_fc3_survival)
                tf.summary.histogram('out_survival' + '/weight', w3_fc_survival)
                tf.summary.histogram('out_survival' + '/bias', b3_fc_survival)
        # show_img = lrn1[:, :, :, 0, np.newaxis]
        # tf.summary.image('03: conv1', show_img, 3)
        #
        # show_img = lrn2[:, :, :, 0, np.newaxis]
        # tf.summary.image('04: conv2', show_img, 3)
        #
        # show_img = lrn3[:, :, :, 0, np.newaxis]
        # tf.summary.image('05: conv3', show_img, 3)
        #
        # show_img = lrn4[:, :, :, 0, np.newaxis]
        # tf.summary.image('06: conv4', show_img, 3)
        #
        # show_img = lrn5[:, :, :, 0, np.newaxis]
        # tf.summary.image('07: conv5', show_img, 3)

        with tf.variable_scope('loss'):
            # alpha=.3
            # corss entropy
            cross_entropy = -tf.reduce_mean(self.labels_grad * tf.log(self.out_grad)) \
                            - tf.reduce_mean(self.labels_survival * tf.log(self.out_survival))
            # cross_entropy = -alpha*(tf.reduce_mean(self.labels_grad*tf.log(self.out_grad)) )\
            #                 -(1-alpha)*(tf.reduce_mean(self.labels_survival*tf.log(self.out_survival)))

            # L2 regularization for the fully connected parameters.
            # regularizers = (
            #                 tf.nn.l2_loss(w2_fc_survival) + tf.nn.l2_loss(b2_fc_survival) +
            #                 tf.nn.l2_loss(w2_fc_grad) + tf.nn.l2_loss(b2_fc_grad) +
            #                 tf.nn.l2_loss(w3_fc_survival) + tf.nn.l2_loss(b3_fc_survival) +
            #                 tf.nn.l2_loss(w3_fc_grad) + tf.nn.l2_loss(b3_fc_grad) +
            #                 tf.nn.l2_loss(w1_fu) + tf.nn.l2_loss(b1_fu) +
            #                 tf.nn.l2_loss(w5_conv) + tf.nn.l2_loss(b5_conv) +
            #                 tf.nn.l2_loss(w4_conv) + tf.nn.l2_loss(b4_conv) +
            #                 tf.nn.l2_loss(w3_conv) + tf.nn.l2_loss(b3_conv) +
            #                 tf.nn.l2_loss(w2_conv) + tf.nn.l2_loss(b2_conv) +
            #                 tf.nn.l2_loss(w1_conv) + tf.nn.l2_loss(b1_conv)
            #                 )
            self.loss = cross_entropy + 5e-4 #*regularizers
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):

            self.learning_rate = tf.train.exponential_decay(
                self.lr,  # Base learning rate.
                #batch * self.batch_size,  # Current index into the dataset.
                self.global_step,
                5000,  # Decay step.
                0.98,  # Decay rate.
                staircase=True)
            tf.summary.scalar('learning rate',self.learning_rate)

            self._train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=self.global_step)

            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss) # normal training
            # self._train_op = tf.train.MomentumOptimizer(self.lr,0.9).minimize(self.loss)
            # learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.global_step,
            #                                            decay_steps=10000, decay_rate=0.96, staircase=True)
            # grad_norm = 8
            # tvars = tf.trainable_variables()
            # grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),grad_norm) # adding clipping
            # opt = tf.train.RMSPropOptimizer(self.lr)
            # self._train_op = opt.apply_gradients(zip(grads,tvars))

    def accuracy(self,predictions, labels):
        """
        Get accuracy
        :param predictions:
        :param labels:
        :return: accuracy
        """
        size = labels.shape[0]
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / size)

    def get_accuracy_rate(self,x,y_grad,y_survival,flag=3,point=0,pre_loss=0):
        cost,pred_grad,pred_survival,rs = self.sess.run([self.loss,self.out_grad,self.out_survival,self.merged],
                                                  feed_dict={
                                                      self.labels_grad: y_grad,
                                                      self.labels_survival: y_survival,
                                                      self.xs: x,
                                                      self.loss_ts:pre_loss,
                                                      self.keep_prob: 0.5

                                                  })
        if flag==1:
            self.train_writer.add_summary(rs, point)
        elif flag==2:
            self.validation_writer.add_summary(rs, point)
        else:
            a=1

        accu_rate_grad = self.accuracy(pred_grad, y_grad)
        accu_rate_survival = self.accuracy(pred_survival, y_survival)
        return cost,accu_rate_grad,accu_rate_survival

    def get_result(self,x):
        """
        :param x:
        :return: predicted survival and gradder
        """
        pred_grad, pred_survival = self.sess.run([self.out_grad, self.out_survival,],feed_dict={self.xs: x})
        grad = np.argmax(pred_grad, 1)
        survival = np.argmax(pred_survival, 1)
        print (pred_grad)
        print (pred_survival)
        return grad,survival


    def learn(self,x,y_grad,y_survival,point):

        # train eval network
        _, self.cost= self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                                self.labels_grad: y_grad,
                                                self.labels_survival: y_survival,
                                                self.xs: x,
                                                self.keep_prob:0.5
                                                })


        self.global_counter +=1
        if self.global_counter%10==0:
            self.cost_his.append(self.cost)

    def plot_cost(self):
        """
        This function will plot cost histgram
        :return:
        """
        import matplotlib.pyplot as plt

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        # plt.show()
        plt.grid()
        plt.savefig('cost.png')


    def restore_parameters(self):
        """
        This function will restore weights
        :return:
        """
        self.saver.restore(self.sess, self.dir_path + '/weights_saved/model.ckpt')  # restore model

    def save_parameters(self):
        """
        This function will save weights
        :return:
        """
        saver = tf.train.Saver()
        if not os.path.exists(self.dir_path+"/weights_saved"):
            os.mkdir(self.dir_path + "/weights_saved")
        saver_path = saver.save(self.sess, self.dir_path+'/weights_saved/model.ckpt')  # save model into save/model.ckpt file
        print('Model saved in file:', saver_path)

    def merge_hist(self,x,y_grad,y_survival,point):
        rs = self.sess.run(self.merged, feed_dict={
                                                self.labels_grad: y_grad,
                                                self.labels_survival: y_survival,
                                                self.xs: x,
                                                self.keep_prob: 0.5,
            self.loss_ts:0})
        self.writer.add_summary(rs,point)
        self.writer.flush()


