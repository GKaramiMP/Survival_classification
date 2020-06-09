
import numpy as np
import os
import sys
import pickle
import random
from joint.net import Network
import matplotlib.pyplot as plt



def train_model():

    train_path = '/exports/lkeb-hpc/gkarami/Data/2_Train_T1/'
    train_dataset = sorted(next(os.walk(train_path))[2])
    train_age_labels =  np.mat(("3; 3; 1; 1; 2; 2; 2;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 2; 2; 2; 1; 2; 2; 2; 1; 1; 1; 2;  2; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 1 "), dtype=float)
    train_gender_lables =  np.mat(("0; 0; 1; 1; 0; 0 0;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 0;  1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1 "), dtype=float)
    
    valid_path = '/exports/lkhpc/gkarami/Data/2_Validation_T1_hg/'
    valid_dataset = sorted(next(os.walk(valid_path))[2])
    valid_age_labels = np.mat((" 1; 1; 1; 0; 0; 0; 2; 2; 1; 1; 3; 3; 2"), dtype=float)
    valid_gender_labels = np.mat((" 1; 1; 1; 0; 0; 0; 1; 0; 1; 1; 0; 0; 1"), dtype=float)
    
    test_path = '/exports/lkeb-hpc/gkarami/Data/2_Test_T1_hg/'
    test_dataset = sorted(next(os.walk(test_path))[2])
    test_age_labels = np.mat((" 1; 1; 0; 0; 0; 0; 0; 2; 2; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 2; 3; 3  "), dtype=float)
    test_gender_labels = np.mat((" 1; 1; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0  "), dtype=float)

    hight = 128
    channel = 1
    batch_size = 128
    learn_rate = 0.001
    n_output_age = 4
    n_output_gen = 2
        
    total_size = train_dataset.shape[0]

    net = Network(
        n_output_gen = n_output_gen,
        n_output_age = n_output_age,
        n_length=hight,
        learning_rate=learn_rate,
        batch_size=batch_size,
        channel=channel,
        output_graph=True,
        use_ckpt=False
    )
    epoch = 400
    iteration = int(total_size / batch_size)

    early_stop = 0 #flag of early stopping
    i=1 # total training time
    accu_train_gen=[]
    accu_valid_gen=[]
    accu_test_gen=[]
    accu_train_age = []
    accu_valid_age = []
    accu_test_age = []
    train_rate_gen,train_rate_age = 0,0

    for e in range(epoch):
        print("-------------------------------")
        print("epoch %d" %(e+1))
        # randomly sample batch memory from all memory
        indices = np.random.permutation(total_size)
        for ite in range(iteration):
            mini_indices = indices[ite*batch_size:(ite+1)*batch_size]
            batch_x = train_dataset[mini_indices, :, :, :]
            batch_y_gen = train_gender_lables[mini_indices, :]
            batch_y_age = train_age_labels[mini_indices, :]
            net.learn(batch_x,batch_y_gen,batch_y_age)

            if i%50==0:
                cost,train_rate_gen,train_rate_age = net.get_accuracy_rate(batch_x,batch_y_gen,batch_y_age)
                print("Iteration: %i. Train loss %.5f, Minibatch gen accuracy:"" %.1f%%,Minibatch age accuracy:"" %.1f%%"
                      % (i, cost, train_rate_gen,train_rate_age))
                accu_train_gen.append(train_rate_gen),accu_train_age.append(train_rate_age)

            if i%50==0:
                cost, valid_rate_gen,valid_rate_age = net.get_accuracy_rate(valid_dataset,valid_gender_labels,valid_age_labels)
                print("Iteration: %i. Validation loss %.5f, Validation gen accuracy:" " %.1f%% ,Validation age accuracy:" " %.1f%%"
                      % (i, cost, valid_rate_gen,valid_rate_age))
                accu_valid_gen.append(valid_rate_gen), accu_valid_age.append(valid_rate_age)
                cost, test_rate_gen,test_rate_age = net.get_accuracy_rate(test_dataset, test_gender_labels,test_age_labels)
                print("Iteration: %i. Test loss %.5f, Test gen accuracy:"" %.1f%%,Test age accuracy:"" %.1f%%"
                      % (i, cost, test_rate_gen,test_rate_age))
                accu_test_gen.append(test_rate_gen), accu_test_age.append(test_rate_age)
            if i%500==0:
                net.save_parameters()

            if i%5==0: # save histogram
                net.merge_hist(batch_x,batch_y_gen,batch_y_age)
            i = i+1

        # early stopping
        if train_rate_gen==100 and train_rate_age==100:
            if early_stop==10:
                print("Early Stopping!")
                break
            else:early_stop = early_stop+1

    net.plot_cost() # plot trainingi cost

    plt.figure()   # plot accuracy
    plt.plot(np.arange(len(accu_train_gen)), accu_train_gen,label='train gender',linestyle='--' )
    plt.plot(np.arange(len(accu_valid_gen)), accu_valid_gen,label='valid gender',linestyle='-')
    plt.plot(np.arange(len(accu_test_gen)), accu_test_gen,label='test gender',linestyle=':')
    plt.ylabel('gender accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('gender.png')

    plt.figure()  # plot accuracy
    plt.plot(np.arange(len(accu_train_age)), accu_train_age,label='train age',linestyle='--')
    plt.plot(np.arange(len(accu_valid_age)), accu_valid_age,label='valid age',linestyle='-')
    plt.plot(np.arange(len(accu_test_age)), accu_test_age,label='test age',linestyle=':')
    plt.ylabel('age accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('age.png')
    # plt.show()


def main():
    train_model()

if __name__=='__main__':
    main()

# =================================================================================================================================================
import numpy as np
import os
import sys
import pickle
import random
import cv2
import self as self
import glob
from skimage.io import imread
from skimage.transform import resize
import tensorflow.python.framework.dtypes
from joint.net import Network
import matplotlib.pyplot as plt
import tensorflow as tf
#from keras.utils import np_utils
# ========================

def train_model():
    # ============================================================ Train setup
    train_path = '/exports/lkeb-hpc/gkarami/Data/2_Train_T1/'
    mask_train_path = '/exports/lkeb-hpc/gkarami/Data/2_Train_T1/'
    train_dataset_1 = sorted(next(os.walk(train_path))[2])
    train_dataset_main = np.zeros((len(train_dataset_1), 128, 128), dtype=np.float)
    Resized_Train = np.zeros((len(train_dataset_1), 64, 64), dtype=np.float)
    Resized_Masks_Train = np.zeros((len(train_dataset_1), 64, 64, 1), dtype=np.bool)

    n = 0
    for mask_path in glob.glob('{}/*.tif'.format(mask_train_path)):
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}.tif'.format(train_path, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        y_coord, x_coord = np.where(mask != 0)

        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        Resized_Train[n] = resize(cropped_image[:, :],
                                     (64, 64),
                                     mode='constant',
                                     anti_aliasing=True,
                                     preserve_range=True)

        Resized_Masks_Train[n] = np.expand_dims(resize(cropped_mask,
                                                       (64, 64),
                                                       mode='constant',
                                                       anti_aliasing=True,
                                                       preserve_range=True), axis=-1)

        train_dataset_main[n] = image

        n += 1

    Rot_90_Train = np.zeros((len(train_dataset_1), 64, 64), dtype=np.float)
    Rot_lr_Train = np.zeros((len(train_dataset_1), 64, 64), dtype=np.float)
    Rot_ud_Train = np.zeros((len(train_dataset_1), 64, 64), dtype=np.float)
    for m in range(len(train_dataset_1)):
        img = Resized_Train[m]
        Rot_90_Train[m] = np.rot90(img)
        Rot_lr_Train[m] = np.fliplr(img)
        Rot_ud_Train[m] = np.flipud(img)

    train_dataset = np.concatenate((Resized_Train, Rot_90_Train, Rot_lr_Train, Rot_ud_Train), axis=0)
    # train_dataset = Resized_Train
    train_dataset = train_dataset[..., np.newaxis]
    train_dataset = tf.keras.utils.normalize(train_dataset, axis=1)

    # ==================================================================== validation setup
    valid_path = '/exports/lkeb-hpc/gkarami/Data/2_Validation_T1/'
    valid_dataset_1 = sorted(next(os.walk(valid_path))[2])
    valid_dataset_main = np.zeros((len(valid_dataset_1), 128, 128), dtype=np.float)
    Resized_Validation = np.zeros((len(valid_dataset_1), 64, 64), dtype=np.float)
    Resized_Masks_Valid = np.zeros((len(valid_dataset_1), 64, 64, 1), dtype=np.bool)

    n = 0
    for mask_path in glob.glob('{}/*.tif'.format(valid_path)):
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}.tif'.format(valid_path, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        y_coord, x_coord = np.where(mask != 0)

        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        Resized_Validation[n] = resize(cropped_image[:, :],
                                          (64, 64),
                                          mode='constant',
                                          anti_aliasing=True,
                                          preserve_range=True)

        Resized_Masks_Valid[n] = np.expand_dims(resize(cropped_mask,
                                                            (64, 64),
                                                            mode='constant',
                                                            anti_aliasing=True,
                                                            preserve_range=True), axis=-1)

        valid_dataset_main[n] = image

        n += 1

    Rot_90_Validation = np.zeros((len(valid_dataset_1), 64, 64), dtype=np.float)
    Rot_lr_Validation = np.zeros((len(valid_dataset_1), 64, 64), dtype=np.float)
    Rot_ud_Validation = np.zeros((len(valid_dataset_1), 64, 64), dtype=np.float)
    for m in range(len(valid_dataset_1)):
        img = Resized_Validation[m]
        Rot_90_Validation[m] = np.rot90(img)
        Rot_lr_Validation[m] = np.fliplr(img)
        Rot_ud_Validation[m] = np.flipud(img)

    valid_dataset = np.concatenate((Resized_Validation, Rot_90_Validation, Rot_lr_Validation, Rot_ud_Validation), axis=0)
    # valid_dataset = Resized_Validation
    valid_dataset = valid_dataset[..., np.newaxis]
    valid_dataset = tf.keras.utils.normalize(valid_dataset, axis=1)


    # ======================================================== Test setup
    test_path = '/exports/lkeb-hpc/gkarami/Data/2_Test_T1/'
    test_dataset_1 = sorted(next(os.walk(test_path))[2])
    test_dataset_main = np.zeros((len(test_dataset_1), 128, 128), dtype=np.float)
    Resized_Test = np.zeros((len(test_dataset_1), 64, 64), dtype=np.float)
    Resized_Masks_Test = np.zeros((len(test_dataset_1), 64, 64, 1), dtype=np.bool)

    n = 0
    for mask_path in glob.glob('{}/*.tif'.format(test_path)):
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}.tif'.format(test_path, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        y_coord, x_coord = np.where(mask != 0)

        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        Resized_Test[n] = resize(cropped_image[:, :],
                                    (64, 64),
                                    mode='constant',
                                    anti_aliasing=True,
                                    preserve_range=True)

        Resized_Masks_Test[n] = np.expand_dims(resize(cropped_mask,
                                                      (64, 64),
                                                      mode='constant',
                                                      anti_aliasing=True,
                                                      preserve_range=True), axis=-1)

        test_dataset_main[n] = image

        n += 1

    Rot_90_Test = np.zeros((len(test_dataset_1), 64, 64), dtype=np.float)
    Rot_lr_Test = np.zeros((len(test_dataset_1), 64, 64), dtype=np.float)
    Rot_ud_Test = np.zeros((len(test_dataset_1), 64, 64), dtype=np.float)
    for m in range(len(test_dataset_1)):
        img = Resized_Test[m]
        Rot_90_Test[m] = np.rot90(img)
        Rot_lr_Test[m] = np.fliplr(img)
        Rot_ud_Test[m] = np.flipud(img)

    test_dataset = np.concatenate((Resized_Test, Rot_90_Test, Rot_lr_Test, Rot_ud_Test), axis=0)
    # test_dataset =Resized_Test
    test_dataset = test_dataset[..., np.newaxis]
    test_dataset = tf.keras.utils.normalize(test_dataset, axis=1)
    # ==================================================================
    train_age_labels = np.mat((   "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                                  "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                                  "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; "
                                  "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2 "), dtype=float)

    # train_age_labels = np.mat((   "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2"), dtype=float)

    train_gender_labels = np.mat((" 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;"
                                  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;"
                                  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;"
                                  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0"), dtype=float)

    # train_gender_labels = np.mat((" 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;  0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0"), dtype= float)

    train_age_labels = (tf.keras.utils.to_categorical(train_age_labels))
    train_gender_labels = (tf.keras.utils.to_categorical(train_gender_labels))

    valid_age_labels = np.mat(( " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2"), dtype=float)

    # valid_age_labels = np.mat((  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2"), dtype=float)

    valid_gender_labels = np.mat((" 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
                                  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
                                  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
                                  " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1"), dtype=float)

    # valid_gender_labels = np.mat((" 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1"), dtype=float)
    valid_gender_labels = (tf.keras.utils.to_categorical(valid_gender_labels))
    valid_age_labels = (tf.keras.utils.to_categorical(valid_age_labels))

    test_age_labels = np.mat(( "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                               "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                               "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
                               "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2"), dtype=float)

    # test_age_labels = np.mat(( "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2"), dtype=float)

    test_gender_labels = np.mat(("1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0"), dtype=float)

    # test_gender_labels = np.mat(("1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0"), dtype=float)
    test_gender_labels = (tf.keras.utils.to_categorical(test_gender_labels))
    test_age_labels = (tf.keras.utils.to_categorical(test_age_labels))

    hight = 128
    channel = 1
    batch_size = 5
    learn_rate = 0.001
    n_output_age = 3
    n_output_gen = 2

    total_size = train_dataset.shape[0]

    net = Network(
        n_output_gen=n_output_gen,
        n_output_age=n_output_age,
        n_length=hight,
        learning_rate=learn_rate,
        batch_size=batch_size,
        channel=channel,
        output_graph=True,
        use_ckpt=False
    )
    epoch = 400
    iteration = int(total_size / batch_size)
    early_stop = 0  # flag of early stopping
    i = 1  # total training time
    accu_train_gen = []
    accu_valid_gen = []
    accu_test_gen = []
    accu_train_age = []
    accu_valid_age = []
    accu_test_age = []
    train_rate_gen, train_rate_age = 0, 0

    for e in range(epoch):
        print("-------------------------------")
        print("epoch %d" % (e + 1))
        # randomly sample batch memory from all memory
        indices = np.random.permutation(total_size)
        for ite in range(iteration):
            mini_indices = indices[ite * batch_size:(ite + 1) * batch_size]
            batch_x = train_dataset[mini_indices, :, :, :]
            batch_y_gen = train_gender_labels[mini_indices, :]
            batch_y_age = train_age_labels[mini_indices, :]
            net.learn(batch_x, batch_y_gen, batch_y_age)

            if i % 5 == 0:
                cost, train_rate_gen, train_rate_age = net.get_accuracy_rate(batch_x, batch_y_gen, batch_y_age)
                print(
                    "Iteration: %i. Train loss %.5f, Minibatch gen accuracy:"" %.1f%%,Minibatch age accuracy:"" %.1f%%"
                    % (i, cost, train_rate_gen, train_rate_age))
                accu_train_gen.append(train_rate_gen), accu_train_age.append(train_rate_age)

            if i % 5 == 0:
                cost, valid_rate_gen, valid_rate_age = net.get_accuracy_rate(valid_dataset, valid_gender_labels,
                                                                             valid_age_labels)
                print(
                    "Iteration: %i. Validation loss %.5f, Validation gen accuracy:" " %.1f%% ,Validation age accuracy:" " %.1f%%"
                    % (i, cost, valid_rate_gen, valid_rate_age))
                accu_valid_gen.append(valid_rate_gen), accu_valid_age.append(valid_rate_age)
                cost, test_rate_gen, test_rate_age = net.get_accuracy_rate(test_dataset, test_gender_labels,
                                                                           test_age_labels)
                print("Iteration: %i. Test loss %.5f, Test gen accuracy:"" %.1f%%,Test age accuracy:"" %.1f%%"
                      % (i, cost, test_rate_gen, test_rate_age))
                accu_test_gen.append(test_rate_gen), accu_test_age.append(test_rate_age)
            if i % 50 == 0:
                net.save_parameters()

            # if i % 5 == 0:  # save histogram
            #     net.merge_hist(batch_x, batch_y_gen, batch_y_age)
            i = i + 1

        # early stopping
        if train_rate_gen == 100 and train_rate_age == 100:
            if early_stop == 10:
                print("Early Stopping!")
                break
            else:
                early_stop = early_stop + 1

    # net.plot_cost()  # plot trainingi cost
    #
    # plt.figure()  # plot accuracy
    # plt.plot(np.arange(len(accu_train_gen)), accu_train_gen, label='train gender', linestyle='--')
    # plt.plot(np.arange(len(accu_valid_gen)), accu_valid_gen, label='valid gender', linestyle='-')
    # plt.plot(np.arange(len(accu_test_gen)), accu_test_gen, label='test gender', linestyle=':')
    # plt.ylabel('gender accuracy')
    # plt.xlabel('epoch')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()
    # #plt.savefig('gender.png')
    #
    # plt.figure()  # plot accuracy
    # plt.plot(np.arange(len(accu_train_age)), accu_train_age, label='train age', linestyle='--')
    # plt.plot(np.arange(len(accu_valid_age)), accu_valid_age, label='valid age', linestyle='-')
    # plt.plot(np.arange(len(accu_test_age)), accu_test_age, label='test age', linestyle=':')
    # plt.ylabel('age accuracy')
    # plt.xlabel('epoch')
    # plt.legend(loc='lower right')
    # plt.grid()
    # #plt.savefig('age.png')
    # plt.show()


def main():
    train_model()


if __name__ == '__main__':
    main()