
'''============================== A CNN Model for classification patients '''

import os

# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import shuffle
from keras.layers.merge import Concatenate
from keras.utils import np_utils
from skimage.io import imread

Path_Validation_T1 = '/exports/lkeb-hpc/gkarami/Data/2_Validation_T1/'
Path_Validation_MD = '/exports/lkeb-hpc/gkarami/Data/2_Validation_MD/'
Path_Validation_CBV = '/exports/lkeb-hpc/gkarami/Data/2_Validation_CBV/'
Path_Validation_Masks = '/exports/lkeb-hpc/gkarami/Data/2_Validation_T1/'




IMG_validation_T1 = sorted(next(os.walk(Path_Validation_T1))[2])
IMG_validation_MD = sorted(next(os.walk(Path_Validation_MD))[2])
IMG_validation_CBV = sorted(next(os.walk(Path_Validation_CBV))[2])

# IMG_test = sorted(next(os.walk(x_test))[2])


IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
IMG_Subjects = 2


Inputs_validation_T1 = np.zeros((len(IMG_validation_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
Inputs_validation_MD = np.zeros((len(IMG_validation_MD), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
Inputs_validation_CBV = np.zeros((len(IMG_validation_CBV), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)


for n, f in enumerate(IMG_validation_T1):
    Images = imread(Path_Validation_T1 + f)[:, :]
    Inputs_validation_T1[n] = Images
for n, f in enumerate(IMG_validation_MD):
    Images = imread(Path_Validation_MD + f)[:, :]
    Inputs_validation_MD[n] = Images
for n, f in enumerate(IMG_validation_CBV):
    Images = imread(Path_Validation_CBV + f)[:, :]
    Inputs_validation_CBV[n] = Images

validation_data = np.stack((Inputs_validation_T1, Inputs_validation_MD, Inputs_validation_CBV), axis=3)

# # =========================================

# validation_T1_reshape  = Inputs_validation_T1.reshape(44, 5, 128, 128, 1)
# validation_MD_reshape  = Inputs_validation_MD.reshape(44, 5, 128, 128, 1)
# validation_CBV_reshape  = Inputs_validation_CBV.reshape(44, 5, 128, 128, 1)

# =========================== for train _T1_hg
y_validation = np.mat((" 0; 0; 0; 0; 0; 2; 2; 2; 1; 1; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1;  1; 1; 2; 2; 2; 0; 0; 0; 2; 2; 1; 1;  1; 1; 0; 0; 0; 0; 0;  1; 1; 1; 1; 2; 2;"
                       " 0; 0; 0; 0; 0; 2; 2; 2; 1; 1; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1;  1; 1; 2; 2; 2; 0; 0; 0; 2; 2; 1; 1;  1; 1; 0; 0; 0; 0; 0;  1; 1; 1; 1; 2; 2"), dtype=float)
y_validation_cat = np_utils.to_categorical(y_validation)

# =========================================== shuffle

def shuffle_lists( T1, MD, CBV):
    index_shuf = list(range(len(T1[0])))
    shuffle(index_shuf)
    T1_sn = np.hstack([T1[sn]]
                             for sn in index_shuf)
    MD_sn = np.hstack([MD[sn]]
                             for sn in index_shuf)
    CBV_sn = np.hstack([CBV[sn]]
                              for sn in index_shuf)
    # y_cat_sn = np.hstack([y_cat[sn]]
    #                                 for sn in index_shuf)
    return T1_sn, MD_sn, CBV_sn

# if __name__=='__main__':


T1_sn, MD_sn, CBV_sn = shuffle_lists( Inputs_validation_T1,  Inputs_validation_MD, Inputs_validation_CBV)

# x_train2 = train[0][1][2]
# y_train2 = train[3]
# ========================================= create the model

input_layer_T1 = Input((5, 128, 128, 1))
input_layer_MD = Input((5, 128, 128, 1))
input_layer_CBV = Input((5, 128, 128, 1))

conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer_T1)
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer1)  # add max pooling to obtain the most imformatic features

conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer1)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer2)  # In 'th' mode, the channels dimension (the depth) is at index 1, in 'tf' mode is it at index 4.

conv_layer3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer2)
pooling_layer3 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer3)

conv_layer4 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer3)
pooling_layer4 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer4)

conv_layer5 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer4)
pooling_layer5 = BatchNormalization()(conv_layer5)
# ============MD

conv_layer11 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer_MD)
pooling_layer11 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer11)  # add max pooling to obtain the most imformatic features

conv_layer22 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer11)
pooling_layer22 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer22)  # In 'th' mode, the channels dimension (the depth) is at index 1, in 'tf' mode is it at index 4.

conv_layer33 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer22)
pooling_layer33 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer33)

conv_layer44 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer33)
pooling_layer44 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer44)

conv_layer55 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer44)
pooling_layer55 = BatchNormalization()(conv_layer55)  # perform batch normalization on the convolution outputs before feeding it to MLP architecture

# ===============CBV

conv_layer111 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer_CBV)
pooling_layer111 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer111)  # add max pooling to obtain the most imformatic features

conv_layer222 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer111)
pooling_layer222 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer222)  # In 'th' mode, the channels dimension (the depth) is at index 1, in 'tf' mode is it at index 4.

conv_layer333 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer222)
pooling_layer333 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer333)

conv_layer444 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer333)
pooling_layer444 = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 2), padding='same', dim_ordering="tf")(conv_layer444)

conv_layer555 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer444)
pooling_layer555 = BatchNormalization()(conv_layer555)  # perform batch normalization on the convolution outputs before feeding it to MLP architecture

# ==============================================
flatten_layer1 = Flatten()(pooling_layer5)  ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
dense_layer1 = (Dense(512, activation='relu'))(flatten_layer1)
dense_layer2 = Dropout(0.4)(dense_layer1)  # add dropouts to avoid overfitting / perform regularization

flatten_layer11 = Flatten()(pooling_layer55)  ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
dense_layer11 = (Dense(512, activation='relu'))(flatten_layer11)
dense_layer22 = Dropout(0.4)(dense_layer11)

flatten_layer111 = Flatten()(pooling_layer555)  ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
dense_layer111 = (Dense(512, activation='relu'))(flatten_layer111)
dense_layer222 = Dropout(0.4)(dense_layer111)
# ===========
dense_layer_con1 = Concatenate()([dense_layer2, dense_layer22, dense_layer222])
dense_layer_con2 = Dense(units=512, activation='relu')(dense_layer_con1)
dense_layer_con3 = Dropout(0.4)(dense_layer_con2)

dense_layer_con4 = Dense(units=256, activation='relu')(dense_layer_con3)
dense_layer_con5 = Dropout(0.4)(dense_layer_con4)

dense_layer_con6 = Dense(units=256, activation='relu')(dense_layer_con5)
dense_layer_con7 = Dropout(0.4)(dense_layer_con6)

output_layer = Dense(units=3, activation='softmax')(dense_layer_con7)

model = Model(inputs=[input_layer_T1, input_layer_MD, input_layer_CBV], outputs=output_layer)  # define the model with input layer and output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # We will measure the performance of the model using accuracy.
model.summary()
validation_norm = [validation_T1_reshape, validation_MD_reshape, validation_CBV_reshape]
model_cnn = model.fit([train_T1_reshape, train_MD_reshape, train_CBV_reshape], y_train_cat, batch_size=100, epochs=150,
                      validation_data=(validation_norm, y_validation_cat))

# ============================
print(model_cnn.history.keys())
loss = model_cnn.history['loss']
val_loss = model_cnn.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss_3D_hg_Aug_3_chan_cat_1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/exports/lkeb-hpc/gkarami/Code/Jobs/loss_3D_hg_Aug_3_chan_cat_1.png')
# plt.show()


acc = model_cnn.history['accuracy']
val_acc = model_cnn.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('accuracy_3D_hg_Aug_3_chan_cat_1')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/exports/lkeb-hpc/gkarami/Code/Jobs/aloss_3D_hg_Aug_3_chan_cat_1.png')

#======================================================================================

import numpy as np
import tensorflow as tf
import os

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class Network(object):
    def __init__(
            self,
            n_output_gen = 2,
            n_volume = 5,
            n_output_survival = 3,
            n_length = 64,
            learning_rate = 0.0001,
            batch_size = 32,
            channel = 3,
            volume = 5,
            output_graph = False,
            use_ckpt = True
    ):


        self.n_length = n_length # width or height of input matrix
        self.n_volume = n_volume
        self.lr = learning_rate
        self.batch_size = batch_size
        self.channel = channel # num of channel
        self.volume = volume,
        self.learn_step_counter = 0
        self.global_step = tf.Variable(tf.constant(1))
        self.global_counter = 1 # equal to self.global_step
        self.global_valid_counter = 1
        self.n_output_survival = n_output_survival
        self.n_output_gen = n_output_gen
        self._build_net()

        # e_params = tf.get_collection('eval_net_params')

        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # self.replace_eval_op = [tf.assign(e, t) for e, t in zip(e_params,t_params)]
        # assign e to t

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        # self.dir_path = '/home/hengtong/project/protein/ProteinHP_DQN/heng_model/save_weight/'
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        self.merged = tf.summary.merge_all()
        if output_graph:
            # tensorboard --logdir=logs
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        if use_ckpt:
            self.restore_parameters()
        else:
            self.sess.run(tf.global_variables_initializer()) # train step

        self.cost_his = []


    def conv3d(self, x, W, stride,pad):
        # stride [v_movment, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv3d( x, W, strides = stride, padding=pad)

    # def max_pool3d(self, x, k, stride, pad ):
    #     # stride [v_movment, x_movement, y_movement, 1]
    #     return tf.nn.max_pool(x, ksize=k, strides=stride, padding=pad)

    def BN_fc(self,x,dim):
        # x is input size,dim is batch size
        mean_value,var_value = tf.nn.moments(x,[0])
        scales = tf.Variable(tf.ones([dim]))
        betas = tf.Variable(tf.zeros([dim]))
        epsilon = 1e-3
        return tf.nn.batch_normalization(x,mean_value,var_value,scales,betas,epsilon)


    def _build_net(self):
        # ------------------ build Gender_Net ------------------
        self.xs = tf.placeholder(tf.float32, [None, self.n_volume, self.n_length, self.n_length, self.channel], name='input')  # input
        # self.s = tf.reshape(self.xs, [-1, self.volume, self.n_features, self.n_features, self.channel])
        # self.keep_prob = tf.placeholder(tf.float32)
        self.labels_gen = tf.placeholder(tf.float32,[None,self.n_output_gen],name='gen_labels')
        self.labels_survival = tf.placeholder(tf.float32, [None, self.n_output_survival], name='survival_labels')

        with tf.variable_scope('multi-para'):
            self.p = tf.Variable(0.5, name='p')
            self.q = tf.Variable(0.5, name='q')
            tf.summary.scalar('multi-para' + 'p', self.p)
            tf.summary.scalar('multi-para' + 'q', self.q)

        with tf.variable_scope('joint_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['gender_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.truncated_normal_initializer(0., 0.1), tf.constant_initializer(0.1)  # config of layers



            # first layer. collections is used later when assign to target net
            with tf.variable_scope('convol1'):
                w1_conv = tf.get_variable('w1_conv', [3, 3, 3, self.channel, 16], initializer=w_initializer, collections=c_names)
                b1_conv = tf.get_variable('b1_conv', [1, 16], initializer=b_initializer, collections=c_names)
                h_conv1 = tf.nn.relu(self.conv3d(self.xs, w1_conv, stride=[1, 1, 1, 1, 1], pad='SAME') + b1_conv)  # output size 64*64*32
                # lrn1 = tf.nn.local_response_normalization(h_conv1, alpha=0.0001, beta=0.75)
                h_pool1 = tf.nn.max_pool3d(h_conv1, k=[ 1, 2, 2, 2, 1], stride=[ 1, 2, 2, 2, 1], pad='SAME')  # output size 32*32*32
                tf.summary.histogram('convol1' + '/kernel', w1_conv)
                tf.summary.histogram('convol1' + '/bias', b1_conv)

             # second layer. collections is used later when assign to target net
            with tf.variable_scope('convol2'):
                w2_conv = tf.get_variable('w2_conv', [3, 3, 3, 16, 16], initializer=w_initializer, collections=c_names)
                b2_conv = tf.get_variable('b2_conv', [1, 16], initializer=b_initializer, collections=c_names)
                h_conv2 = tf.nn.relu(self.conv3d(h_pool1, w2_conv, stride=[1, 1, 1, 1, 1], pad='SAME') + b2_conv)  # output size 32*32*64
                # lrn2 = tf.nn.local_response_normalization(h_conv2, alpha=0.0001, beta=0.75)
                h_pool2 = self.max_pool3d(h_conv2, k=[1, 2, 2, 2, 1], stride=[1, 2, 2, 2, 1], pad='SAME')  # output 16*16*64
                tf.summary.histogram('convol2' + '/kernel', w2_conv)
                tf.summary.histogram('convol2' + '/bias', b2_conv)

            with tf.variable_scope('convol3'):
                w3_conv = tf.get_variable('w3_conv', [3, 3, 3, 16, 32], initializer=w_initializer, collections=c_names)
                b3_conv = tf.get_variable('b3_conv', [1, 32], initializer=b_initializer, collections=c_names)
                h_conv3 = tf.nn.relu(self.conv3d(h_pool2, w3_conv, stride=[1, 1, 1, 1, 1], pad='SAME') + b3_conv)  # output size 16*16*64
                # lrn3 = tf.nn.local_response_normalization(h_conv3, alpha=0.0001, beta=0.75)
                h_pool3 = self.max_pool3d(h_conv3, k=[1, 2, 2, 2, 1], stride=[1, 2, 2, 2, 1], pad='SAME')  # output 8*8*64
                tf.summary.histogram('convol3' + '/kernel', w3_conv)
                tf.summary.histogram('convol3' + '/bias', b3_conv)


            # fully connected layer 1
            with tf.variable_scope('fcl1'):
                w1_fu = tf.get_variable('w1_fu', [2 * 8 * 8 * 32, 512], initializer=w_initializer, collections=c_names)
                b1_fu = tf.get_variable('b1_fu', [1, 512], initializer=b_initializer, collections=c_names)
                h_pool3_flat = tf.reshape(h_pool3, [-1, 2 * 8 * 8 * 32])
                bn_in_fc1 = tf.matmul(h_pool3_flat, w1_fu) + b1_fu
                # bn_out_fc1 = self.BN_fc(bn_in_fc1,512)
                h_fc1 = tf.nn.relu(bn_in_fc1)
                # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
                tf.summary.histogram('fcl1' + '/weight', w1_fu)
                tf.summary.histogram('fcl1' + '/bias', b1_fu)

            # fully connected layer 2,gender
            with tf.variable_scope('fcl2_gen'):
                w2_fc_gen = tf.get_variable('w2_fc_gen', [512, 100], initializer=w_initializer, collections=c_names)
                b2_fc_gen = tf.get_variable('b2_fc_gen', [1, 100], initializer=b_initializer, collections=c_names)
                bn_fc2_gen = tf.matmul(h_fc1, w2_fc_gen) + b2_fc_gen
                # self.q_eval = self.BN_fc(bn_in_fc2, self.n_actions)
                h_fc2_gen = tf.nn.relu(bn_fc2_gen)
                tf.summary.histogram('fcl2_gen' + '/weight', w2_fc_gen)
                tf.summary.histogram('fcl2_gen' + '/bias', b2_fc_gen)

            # fully connected layer 2,survival
            with tf.variable_scope('fcl2_survival'):
                w2_fc_survival = tf.get_variable('w2_fc_survival', [512, 100], initializer=w_initializer, collections=c_names)
                b2_fc_survival = tf.get_variable('b2_fc_survival', [1, 100], initializer=b_initializer, collections=c_names)
                bn_fc2_survival = tf.matmul(h_fc1, w2_fc_survival) + b2_fc_survival
                h_fc2_survival = tf.nn.relu(bn_fc2_survival)
                tf.summary.histogram('fcl2_survival' + '/weight', w2_fc_survival)
                tf.summary.histogram('fcl2_survival' + '/bias', b2_fc_survival)

            # output layer,gender
            with tf.variable_scope('out_gen'):
                w3_fc_gen = tf.get_variable('w3_fc_gen', [100, self.n_output_gen], initializer=w_initializer, collections=c_names)
                b3_fc_gen = tf.get_variable('b3_fc_gen', [1, self.n_output_gen], initializer=b_initializer, collections=c_names)
                bn_fc3_gen = tf.matmul(h_fc2_gen, w3_fc_gen) + b3_fc_gen
                # self.q_eval = self.BN_fc(bn_in_fc2, self.n_actions)
                self.out_gen = tf.multiply(self.q,tf.nn.softmax(bn_fc3_gen))
                tf.summary.histogram('out_gen' + '/weight', w3_fc_gen)
                tf.summary.histogram('out_gen' + '/bias', b3_fc_gen)

            # output layer,survival
            with tf.variable_scope('out_survival'):
                w3_fc_survival = tf.get_variable('w3_fc_survival', [100, self.n_output_survival], initializer=w_initializer, collections=c_names)
                b3_fc_survival = tf.get_variable('b3_fc_survival', [1, self.n_output_survival], initializer=b_initializer, collections=c_names)
                bn_fc2_survival = tf.matmul(h_fc2_survival, w3_fc_survival) + b3_fc_survival
                self.out_survival = tf.multiply(self.p, tf.nn.softmax(bn_fc2_survival))
                tf.summary.histogram('out_survival' + '/weight', w3_fc_survival)
                tf.summary.histogram('out_survival' + '/bias', b3_fc_survival)

        with tf.variable_scope('loss'):

            # corss entropy
            cross_entropy = -tf.reduce_mean(self.labels_gen*tf.log(self.out_gen)) \
                            -tf.reduce_mean(self.labels_survival*tf.log(self.out_survival))

            # cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_gen,logits=self.out_gen)+  ##############################
            #               tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_survival, logits=self.out_survival))

            # L2 regularization for the fully connected parameters.
            regularizers = (
                            tf.nn.l2_loss(w2_fc_survival) + tf.nn.l2_loss(b2_fc_survival) +
                            tf.nn.l2_loss(w2_fc_gen) + tf.nn.l2_loss(b2_fc_gen) +
                            tf.nn.l2_loss(w3_fc_survival) + tf.nn.l2_loss(b3_fc_survival) +
                            tf.nn.l2_loss(w3_fc_gen) + tf.nn.l2_loss(b3_fc_gen) +
                            tf.nn.l2_loss(w1_fu) + tf.nn.l2_loss(b1_fu) +
                            tf.nn.l2_loss(w3_conv) + tf.nn.l2_loss(b3_conv) +
                            tf.nn.l2_loss(w2_conv) + tf.nn.l2_loss(b2_conv) +
                            tf.nn.l2_loss(w1_conv) + tf.nn.l2_loss(b1_conv)
                            )
            self.loss = cross_entropy + 5e-4 *regularizers
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

    def get_accuracy_rate(self,x,y_gen,y_survival):
        cost,pred_gen,pred_survival = self.sess.run([self.loss,self.out_gen,self.out_survival],
                                                  feed_dict={
                                                      self.labels_gen: y_gen,
                                                      self.labels_survival: y_survival,
                                                      self.xs: x
                                                  })
        accu_rate_gen = self.accuracy(pred_gen, y_gen)
        accu_rate_survival = self.accuracy(pred_survival, y_survival)
        return cost,accu_rate_gen,accu_rate_survival

    def get_result(self,x):
        """
        :param x:
        :return: predicted survival and gender
        """
        pred_gen, pred_survival = self.sess.run([self.out_gen, self.out_survival],feed_dict={self.xs: x})
        gen = np.argmax(pred_gen, 1)
        survival = np.argmax(pred_survival, 1)
        print (pred_gen)
        print (pred_survival)
        return gen,survival


    def learn(self,x,y_gen,y_survival):

        # train eval network
        _, self.cost= self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                                self.labels_gen: y_gen,
                                                self.labels_survival: y_survival,
                                                self.xs: x
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

    def merge_hist(self,x,y_gen,y_survival):
        rs = self.sess.run(self.merged, feed_dict={
                                                self.labels_gen: y_gen,
                                                self.labels_survival: y_survival,
                                                self.xs: x})
        self.writer.add_summary(rs)


