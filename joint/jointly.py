
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import glob
import tensorflow as tf
# from keras.losses import binary_crossentropy
import sys
import pickle
import random
import self
from joint.net import Network
from random import shuffle

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from shutil import copyfile


def accuracy(predictions, labels):
    """
    Get accuracy
    :param predictions:
    :param labels:
    :return: accuracy
    """

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy

    # return  tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predictions,1) , tf.argmax(labels,1)), tf.float16))
    #
    # return (100.0 *tf.reduce_sum(tf.cast(tf.argmax(predictions, 1) == tf.argmax(labels, 1),tf.float16))
    #  / tf.cast(tf.shape(labels)[0],tf.float16))
    # size = labels.shape[0]
    # return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    #         / size)


# ================================================
def train_model():
    # ==================================================== train

    train_path_T1 = '/exports/lkeb-hpc/gkarami/Data/3_Train_T1_hg/'
    train_path_MD = '/exports/lkeb-hpc/gkarami/Data/3_Train_MD_hg/'
    train_path_CBV = '/exports/lkeb-hpc/gkarami/Data/3_Train_CBV_hg/'
    train_path_mask = '/exports/lkeb-hpc/gkarami/Data/3_Train_T1_hg/'

    train_dataset_T1_1= sorted(next(os.walk(train_path_T1))[2])
    train_dataset_MD_1 = sorted(next(os.walk(train_path_MD))[2])
    train_dataset_CBV_1= sorted(next(os.walk(train_path_CBV))[2])


    train_dataset_T1 = np.zeros((len(train_dataset_T1_1), 128, 128), dtype=np.float)
    train_dataset_MD = np.zeros((len(train_dataset_MD_1), 128, 128), dtype=np.float)
    train_dataset_CBV = np.zeros((len(train_dataset_CBV_1), 128, 128), dtype=np.float)

    Resized_Train_T1 = np.zeros((len(train_dataset_T1_1), 64, 64), dtype=np.float)
    Resized_Train_MD = np.zeros((len(train_dataset_MD_1), 64, 64), dtype=np.float)
    Resized_Train_CBV = np.zeros((len(train_dataset_CBV_1), 64, 64), dtype=np.float)
    Resized_Train_Masks = np.zeros((len(train_dataset_T1_1), 64, 64, 1), dtype=np.bool)
    print(train_dataset_T1_1)
    #======================================================================valid
    valid_path_T1 = '/exports/lkeb-hpc/gkarami/Data/3_Validation_T1_hg/'
    valid_path_MD = '/exports/lkeb-hpc/gkarami/Data/3_Validation_MD_hg/'
    valid_path_CBV = '/exports/lkeb-hpc/gkarami/Data/3_Validation_CBV_hg/'
    valid_path_mask = '/exports/lkeb-hpc/gkarami/Data/3_Validation_T1_hg/'

    valid_dataset_T1_1 = sorted(next(os.walk(valid_path_T1))[2])
    valid_dataset_MD_1 = sorted(next(os.walk(valid_path_MD))[2])
    valid_dataset_CBV_1 = sorted(next(os.walk(valid_path_CBV))[2])

    valid_dataset_T1 = np.zeros((len(valid_dataset_T1_1), 128, 128), dtype=np.float)
    valid_dataset_MD = np.zeros((len(valid_dataset_MD_1), 128, 128), dtype=np.float)
    valid_dataset_CBV = np.zeros((len(valid_dataset_CBV_1), 128, 128), dtype=np.float)
    valid_dataset_masks = np.zeros((len(valid_dataset_T1_1), 128, 128), dtype=np.bool)

    Resized_Valid_T1 = np.zeros((len(valid_dataset_T1_1), 64, 64), dtype=np.float)
    Resized_Valid_MD = np.zeros((len(valid_dataset_MD_1), 64, 64), dtype=np.float)
    Resized_Valid_CBV = np.zeros((len(valid_dataset_CBV_1), 64, 64), dtype=np.float)
    Resized_Valid_Masks = np.zeros((len(valid_dataset_T1_1), 64, 64, 1), dtype=np.bool)
    print(valid_dataset_T1_1)

    #======================================================================test
    test_path_T1 = '/exports/lkeb-hpc/gkarami/Data/3_Test_T1_hg/'
    test_path_MD = '/exports/lkeb-hpc/gkarami/Data/3_Test_MD_hg/'
    test_path_CBV = '/exports/lkeb-hpc/gkarami/Data/3_Test_CBV_hg/'
    test_path_mask = '/exports/lkeb-hpc/gkarami/Data/3_Test_T1_hg/'

    test_dataset_T1_1 = sorted(next(os.walk(test_path_T1))[2])
    test_dataset_MD_1 = sorted(next(os.walk(test_path_MD))[2])
    test_dataset_CBV_1 = sorted(next(os.walk(test_path_CBV))[2])

    test_dataset_T1 = np.zeros((len(test_dataset_T1_1), 128, 128), dtype=np.float)
    test_dataset_MD = np.zeros((len(test_dataset_MD_1), 128, 128), dtype=np.float)
    test_dataset_CBV = np.zeros((len(test_dataset_CBV_1), 128, 128), dtype=np.float)
    test_dataset_masks = np.zeros((len(test_dataset_T1_1), 128, 128), dtype=np.bool)

    Resized_Test_T1 = np.zeros((len(test_dataset_T1_1), 64, 64), dtype=np.float)
    Resized_Test_MD = np.zeros((len(test_dataset_MD_1), 64, 64), dtype=np.float)
    Resized_Test_CBV = np.zeros((len(test_dataset_CBV_1), 64, 64), dtype=np.float)
    Resized_Test_Masks = np.zeros((len(test_dataset_T1_1), 64, 64, 1), dtype=np.bool)
    print(test_dataset_T1_1)
    # =====================================================================================T1 train
    n = 0
    for mask_path in glob.glob('{}/*.tif'.format(train_path_mask)):
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}.tif'.format(train_path_T1, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        y_coord, x_coord = np.where(mask != 0)

        y_min = min(y_coord)

        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        Resized_Train_T1[n] = resize(cropped_image[:, :],
                                  (64, 64),
                                  mode='constant',
                                  anti_aliasing=True,
                                  preserve_range=True)

        Resized_Train_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                       (64, 64),
                                                       mode='constant',
                                                       anti_aliasing=True,
                                                       preserve_range=True), axis=-1)

        train_dataset_T1[n] = image
        n += 1

    Rot_90_Train_T1 = np.zeros((len(train_dataset_T1_1), 64, 64), dtype=np.float)
    Rot_lr_Train_T1 = np.zeros((len(train_dataset_T1_1), 64, 64), dtype=np.float)
    Rot_ud_Train_T1 = np.zeros((len(train_dataset_T1_1), 64, 64), dtype=np.float)
    for m in range(len(train_dataset_T1_1)):
        img = Resized_Train_T1[m]
        Rot_90_Train_T1[m] = np.rot90(img)
        Rot_lr_Train_T1[m] = np.fliplr(img)
        Rot_ud_Train_T1[m] = np.flipud(img)


    # ===================================================================== CBV  Train
    # n = 0
    # for mask_path in glob.glob('{}/*.tif'.format(train_path_mask)):
    #     base = os.path.basename(mask_path)
    #     image_ID, ext = os.path.splitext(base)
    #     image_path = '{}/{}.tif'.format(train_path_CBV, image_ID)
    #     mask = imread(mask_path)
    #     image = imread(image_path)
    #
    #     y_coord, x_coord = np.where(mask != 0)
    #
    #     y_min = min(y_coord)
    #     y_max = max(y_coord)
    #     x_min = min(x_coord)
    #     x_max = max(x_coord)
    #
    #     cropped_image = image[y_min:y_max, x_min:x_max]
    #     cropped_mask = mask[y_min:y_max, x_min:x_max]
    #
    #     Resized_Train_CBV[n] = resize(cropped_image[:, :],
    #                                   (64, 64),
    #                                   mode='constant',
    #                                   anti_aliasing=True,
    #                                   preserve_range=True)
    #
    #     Resized_Train_Masks[n] = np.expand_dims(resize(cropped_mask,
    #                                                    (64, 64),
    #                                                    mode='constant',
    #                                                    anti_aliasing=True,
    #                                                    preserve_range=True), axis=-1)
    #
    #     train_dataset_CBV[n] = image
    #     n += 1
    #
    # Rot_90_Train_CBV = np.zeros((len(train_dataset_CBV_1), 64, 64), dtype=np.float)
    # Rot_lr_Train_CBV = np.zeros((len(train_dataset_CBV_1), 64, 64), dtype=np.float)
    # Rot_ud_Train_CBV = np.zeros((len(train_dataset_CBV_1), 64, 64), dtype=np.float)
    # for m in range(len(train_dataset_CBV_1)):
    #     img = Resized_Train_CBV[m]
    #     Rot_90_Train_CBV[m] = np.rot90(img)
    #     Rot_lr_Train_CBV[m] = np.fliplr(img)
    #     Rot_ud_Train_CBV[m] = np.flipud(img)
    # # ============================================================ MD  Train
    # n = 0
    # for mask_path in glob.glob('{}/*.tif'.format(train_path_mask)):
    #     base = os.path.basename(mask_path)
    #     image_ID, ext = os.path.splitext(base)
    #     image_path = '{}/{}.tif'.format(train_path_MD, image_ID)
    #     mask = imread(mask_path)
    #     image = imread(image_path)
    #
    #     y_coord, x_coord = np.where(mask != 0)
    #
    #     y_min = min(y_coord)
    #     y_max = max(y_coord)
    #     x_min = min(x_coord)
    #     x_max = max(x_coord)
    #
    #     cropped_image = image[y_min:y_max, x_min:x_max]
    #     cropped_mask = mask[y_min:y_max, x_min:x_max]
    #
    #     Resized_Train_MD[n] = resize(cropped_image[:, :],
    #                                  (64, 64),
    #                                  mode='constant',
    #                                  anti_aliasing=True,
    #                                  preserve_range=True)
    #
    #     Resized_Train_Masks[n] = np.expand_dims(resize(cropped_mask,
    #                                                    (64, 64),
    #                                                    mode='constant',
    #                                                    anti_aliasing=True,
    #                                                    preserve_range=True), axis=-1)
    #
    #     train_dataset_MD[n] = image
    #     n += 1
    #
    # Rot_90_Train_MD = np.zeros((len(train_dataset_MD_1), 64, 64), dtype=np.float)
    # Rot_lr_Train_MD = np.zeros((len(train_dataset_MD_1), 64, 64), dtype=np.float)
    # Rot_ud_Train_MD = np.zeros((len(train_dataset_MD_1), 64, 64), dtype=np.float)
    # for m in range(len(train_dataset_MD_1)):
    #     img = Resized_Train_MD[m]
    #     Rot_90_Train_MD[m] = np.rot90(img)
    #     Rot_lr_Train_MD[m] = np.fliplr(img)
    #     Rot_ud_Train_MD[m] = np.flipud(img)

# ==========================================================================================validataion data
    n = 0
    for mask_path in glob.glob('{}/*.tif'.format(valid_path_mask)):
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}.tif'.format(valid_path_T1, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        y_coord, x_coord = np.where(mask != 0)

        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        Resized_Valid_T1[n] = resize(cropped_image[:, :],
                                     (64, 64),
                                     mode='constant',
                                     anti_aliasing=True,
                                     preserve_range=True)

        Resized_Valid_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                       (64, 64),
                                                       mode='constant',
                                                       anti_aliasing=True,
                                                       preserve_range=True), axis=-1)

        valid_dataset_T1[n] = image
        n += 1

# ======================augmenation
    Rot_90_Valid_T1 = np.zeros((len(valid_dataset_T1_1), 64, 64), dtype=np.float)
    Rot_lr_Valid_T1 = np.zeros((len(valid_dataset_T1_1), 64, 64), dtype=np.float)
    Rot_ud_Valid_T1 = np.zeros((len(valid_dataset_T1_1), 64, 64), dtype=np.float)
    for m in range(len(valid_dataset_T1_1)):
        img = Resized_Valid_T1[m]
        Rot_90_Valid_T1[m] = np.rot90(img)
        Rot_lr_Valid_T1[m] = np.fliplr(img)
        Rot_ud_Valid_T1[m] = np.flipud(img)
#=========================================================================== cbv validation
#     n = 0
#     for mask_path in glob.glob('{}/*.tif'.format(valid_path_mask)):
#         base = os.path.basename(mask_path)
#         image_ID, ext = os.path.splitext(base)
#         image_path = '{}/{}.tif'.format(valid_path_CBV, image_ID)
#         mask = imread(mask_path)
#         image = imread(image_path)
#
#         y_coord, x_coord = np.where(mask != 0)
#
#         y_min = min(y_coord)
#         y_max = max(y_coord)
#         x_min = min(x_coord)
#         x_max = max(x_coord)
#
#         cropped_image = image[y_min:y_max, x_min:x_max]
#         cropped_mask = mask[y_min:y_max, x_min:x_max]
#
#         Resized_Valid_CBV[n] = resize(cropped_image[:, :],
#                                      (64, 64),
#                                      mode='constant',
#                                      anti_aliasing=True,
#                                      preserve_range=True)
#
#         Resized_Valid_Masks[n] = np.expand_dims(resize(cropped_mask,
#                                                        (64, 64),
#                                                        mode='constant',
#                                                        anti_aliasing=True,
#                                                        preserve_range=True), axis=-1)
#
#         valid_dataset_CBV[n] = image
#         n += 1
#
# #=========augmentaion
#     Rot_90_Valid_CBV = np.zeros((len(valid_dataset_CBV_1), 64, 64), dtype=np.float)
#     Rot_lr_Valid_CBV = np.zeros((len(valid_dataset_CBV_1), 64, 64), dtype=np.float)
#     Rot_ud_Valid_CBV = np.zeros((len(valid_dataset_CBV_1), 64, 64), dtype=np.float)
#     for m in range(len(valid_dataset_CBV_1)):
#         img = Resized_Valid_CBV[m]
#         Rot_90_Valid_CBV[m] = np.rot90(img)
#         Rot_lr_Valid_CBV[m] = np.fliplr(img)
#         Rot_ud_Valid_CBV[m] = np.flipud(img)
#
# # ======================================================================  MD  validation
#     n = 0
#     for mask_path in glob.glob('{}/*.tif'.format(valid_path_mask)):
#         base = os.path.basename(mask_path)
#         image_ID, ext = os.path.splitext(base)
#         image_path = '{}/{}.tif'.format(valid_path_MD, image_ID)
#         mask = imread(mask_path)
#         image = imread(image_path)
#
#         y_coord, x_coord = np.where(mask != 0)
#
#         y_min = min(y_coord)
#         y_max = max(y_coord)
#         x_min = min(x_coord)
#         x_max = max(x_coord)
#
#         cropped_image = image[y_min:y_max, x_min:x_max]
#         cropped_mask = mask[y_min:y_max, x_min:x_max]
#
#         Resized_Valid_MD[n] = resize(cropped_image[:, :],
#                                      (64, 64),
#                                      mode='constant',
#                                      anti_aliasing=True,
#                                      preserve_range=True)
#
#         Resized_Valid_Masks[n] = np.expand_dims(resize(cropped_mask,
#                                                        (64, 64),
#                                                        mode='constant',
#                                                        anti_aliasing=True,
#                                                        preserve_range=True), axis=-1)
#
#         valid_dataset_MD[n] = image
#         valid_dataset_masks[n] = mask
#
#         n += 1
# # ===========augmentaion
#     Rot_90_Valid_MD = np.zeros((len(valid_dataset_MD_1), 64, 64), dtype=np.float)
#     Rot_lr_Valid_MD = np.zeros((len(valid_dataset_MD_1), 64, 64), dtype=np.float)
#     Rot_ud_Valid_MD = np.zeros((len(valid_dataset_MD_1), 64, 64), dtype=np.float)
#     for m in range(len(valid_dataset_MD_1)):
#         img = Resized_Valid_MD[m]
#         Rot_90_Valid_MD[m] = np.rot90(img)
#         Rot_lr_Valid_MD[m] = np.fliplr(img)
#         Rot_ud_Valid_MD[m] = np.flipud(img)
# # ========================================================================= =============================test dat
    n = 0
    for mask_path in glob.glob('{}/*.tif'.format(test_path_mask)):
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}.tif'.format(test_path_T1, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)

        y_coord, x_coord = np.where(mask != 0)

        y_min = min(y_coord)
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        Resized_Test_T1[n] = resize(cropped_image[:, :],
                                    (64, 64),
                                    mode='constant',
                                    anti_aliasing=True,
                                    preserve_range=True)

        Resized_Test_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                      (64, 64),
                                                      mode='constant',
                                                      anti_aliasing=True,
                                                      preserve_range=True), axis=-1)

        test_dataset_T1[n] = image
        test_dataset_masks[n] = mask
        n += 1

#=================augmenation
    Rot_90_Test_T1 = np.zeros((len(test_dataset_T1_1), 64, 64), dtype=np.float)
    Rot_lr_Test_T1 = np.zeros((len(test_dataset_T1_1), 64, 64), dtype=np.float)
    Rot_ud_Test_T1 = np.zeros((len(test_dataset_T1_1), 64, 64), dtype=np.float)
    for m in range(len(test_dataset_T1_1)):
        img = Resized_Test_T1[m]
        Rot_90_Test_T1[m] = np.rot90(img)
        Rot_lr_Test_T1[m] = np.fliplr(img)
        Rot_ud_Test_T1[m] = np.flipud(img)

# # ============================================================ for  CBV  Test
#     n = 0
#     for mask_path in glob.glob('{}/*.tif'.format(test_path_mask)):
#         base = os.path.basename(mask_path)
#         image_ID, ext = os.path.splitext(base)
#         image_path = '{}/{}.tif'.format(test_path_CBV, image_ID)
#         mask = imread(mask_path)
#         image = imread(image_path)
#
#         y_coord, x_coord = np.where(mask != 0)
#
#         y_min = min(y_coord)
#         y_max = max(y_coord)
#         x_min = min(x_coord)
#         x_max = max(x_coord)
#
#         cropped_image = image[y_min:y_max, x_min:x_max]
#         cropped_mask = mask[y_min:y_max, x_min:x_max]
#
#         Resized_Test_CBV[n] = resize(cropped_image[:, :],
#                                      (64, 64),
#                                      mode='constant',
#                                      anti_aliasing=True,
#                                      preserve_range=True)
#
#         Resized_Test_Masks[n] = np.expand_dims(resize(cropped_mask,
#                                                       (64, 64),
#                                                       mode='constant',
#                                                       anti_aliasing=True,
#                                                       preserve_range=True), axis=-1)
#
#         test_dataset_CBV[n] = image
#         n += 1
#
# # =================augmenation
#     Rot_90_Test_CBV = np.zeros((len(test_dataset_CBV_1), 64, 64), dtype=np.float)
#     Rot_lr_Test_CBV = np.zeros((len(test_dataset_CBV_1), 64, 64), dtype=np.float)
#     Rot_ud_Test_CBV = np.zeros((len(test_dataset_CBV_1), 64, 64), dtype=np.float)
#     for m in range(len(test_dataset_CBV_1)):
#         img = Resized_Test_CBV[m]
#         Rot_90_Test_CBV[m] = np.rot90(img)
#         Rot_lr_Test_CBV[m] = np.fliplr(img)
#         Rot_ud_Test_CBV[m] = np.flipud(img)
# # =========================================================for MD Test
#     n = 0
#     for mask_path in glob.glob('{}/*.tif'.format(test_path_mask)):
#         base = os.path.basename(mask_path)
#         image_ID, ext = os.path.splitext(base)
#         image_path = '{}/{}.tif'.format(test_path_MD, image_ID)
#         mask = imread(mask_path)
#         image = imread(image_path)
#
#         y_coord, x_coord = np.where(mask != 0)
#
#         y_min = min(y_coord)
#         y_max = max(y_coord)
#         x_min = min(x_coord)
#         x_max = max(x_coord)
#
#         cropped_image = image[y_min:y_max, x_min:x_max]
#         cropped_mask = mask[y_min:y_max, x_min:x_max]
#
#         Resized_Test_MD[n] = resize(cropped_image[:, :],
#                                     (64, 64),
#                                     mode='constant',
#                                     anti_aliasing=True,
#                                     preserve_range=True)
#
#         Resized_Test_Masks[n] = np.expand_dims(resize(cropped_mask,
#                                                       (64, 64),
#                                                       mode='constant',
#                                                       anti_aliasing=True,
#                                                       preserve_range=True), axis=-1)
#
#         test_dataset_MD[n] = image
#         n += 1
#
# # =================augmenation
#     Rot_90_Test_MD = np.zeros((len(test_dataset_MD_1), 64, 64), dtype=np.float)
#     Rot_lr_Test_MD = np.zeros((len(test_dataset_MD_1), 64, 64), dtype=np.float)
#     Rot_ud_Test_MD = np.zeros((len(test_dataset_MD_1), 64, 64), dtype=np.float)
#     for m in range(len(test_dataset_MD_1)):
#         img = Resized_Test_MD[m]
#         Rot_90_Test_MD[m] = np.rot90(img)
#         Rot_lr_Test_MD[m] = np.fliplr(img)
#         Rot_ud_Test_MD[m] = np.flipud(img)
 # ============================================================================================================folder 6_train
    # 02,05,08,09,10,11,12,13, 14,15,16,18,                                                                                                                    19,20,21,22,23,24,25,26,27,28,30,                                                                                                                      31,32 ,34,35,38,40,41,43,46,47,48,
    #
    # train_survival_labels = np.mat(
    #     ("1; 1; 1;      2; 2; 2;    1; 1; 1; 1;    2;   2; 2; 2;        0; 0; 0;    1;      1; 1;        2; 2;      0; 0; 0;        0; 0; 0;       1; 1; 1;     1; 1; 1;    0; 0;    2; 2;   1; 1; 1;    0; 0; 0; 0;     2; 2; 2; 2;     0; 0; 0; 0; 0; 0; 0;        2; 2; 2; 2; 2;     0; 0; 0;     2;   1; 1; 1;    1; 1; 1; 1; 1;        1; 1;       2; 2; 2;      1;      1; 1; 1;    1;     1; 1; 1; 1; 1; 1;           1; 1; 1; 1;         2; 2;          0;   0;"
    #      "1; 1; 1;      2; 2; 2;    1; 1; 1; 1;    2;   2; 2; 2;        0; 0; 0;    1;      1; 1;        2; 2;      0; 0; 0;        0; 0; 0;       1; 1; 1;     1; 1; 1;    0; 0;    2; 2;   1; 1; 1;    0; 0; 0; 0;     2; 2; 2; 2;     0; 0; 0; 0; 0; 0; 0;        2; 2; 2; 2; 2;     0; 0; 0;     2;   1; 1; 1;    1; 1; 1; 1; 1;        1; 1;       2; 2; 2;      1;      1; 1; 1;    1;     1; 1; 1; 1; 1; 1;           1; 1; 1; 1;         2; 2;          0;   0;"
    #      "1; 1; 1;      2; 2; 2;    1; 1; 1; 1;    2;   2; 2; 2;        0; 0; 0;    1;      1; 1;        2; 2;      0; 0; 0;        0; 0; 0;       1; 1; 1;     1; 1; 1;    0; 0;    2; 2;   1; 1; 1;    0; 0; 0; 0;     2; 2; 2; 2;     0; 0; 0; 0; 0; 0; 0;        2; 2; 2; 2; 2;     0; 0; 0;     2;   1; 1; 1;    1; 1; 1; 1; 1;        1; 1;       2; 2; 2;      1;      1; 1; 1;    1;     1; 1; 1; 1; 1; 1;           1; 1; 1; 1;         2; 2;          0;   0;"
    #      "1; 1; 1;      2; 2; 2;    1; 1; 1; 1;    2;   2; 2; 2;        0; 0; 0;    1;      1; 1;        2; 2;      0; 0; 0;        0; 0; 0;       1; 1; 1;     1; 1; 1;    0; 0;    2; 2;   1; 1; 1;    0; 0; 0; 0;     2; 2; 2; 2;     0; 0; 0; 0; 0; 0; 0;        2; 2; 2; 2; 2;     0; 0; 0;     2;   1; 1; 1;    1; 1; 1; 1; 1;        1; 1;       2; 2; 2;      1;      1; 1; 1;    1;     1; 1; 1; 1; 1; 1;           1; 1; 1; 1;         2; 2;          0;   0 " ),
    #     dtype=float)
    #
    # train_grad_labels = np.mat(
    #     ("1; 1; 1;      0; 0; 0;    0; 0; 0; 0;   0;   1; 1; 1;        1; 1; 1;    1;       1; 1;        0; 0;      0; 0; 0;        1; 1; 1;       0; 0; 0;      0; 0; 0;   1; 1;   1; 1;   1; 1; 1;    1; 1; 1; 1;      0; 0; 0; 0;     1; 1; 1; 1; 1; 1; 1;        0; 0; 0; 0; 0;     1; 1; 1;    0;   1; 1; 1;    1; 1; 1; 1; 1;        0; 0;       1; 1; 1;       1;      0; 0; 0;    1;     0; 0; 0; 0; 0; 0;          1; 1; 1; 1;         1; 1;            1;   1;"
    #      "1; 1; 1;      0; 0; 0;    0; 0; 0; 0;   0;   1; 1; 1;        1; 1; 1;    1;       1; 1;        0; 0;      0; 0; 0;        1; 1; 1;       0; 0; 0;      0; 0; 0;   1; 1;   1; 1;   1; 1; 1;    1; 1; 1; 1;      0; 0; 0; 0;     1; 1; 1; 1; 1; 1; 1;        0; 0; 0; 0; 0;     1; 1; 1;    0;   1; 1; 1;    1; 1; 1; 1; 1;        0; 0;       1; 1; 1;       1;      0; 0; 0;    1;     0; 0; 0; 0; 0; 0;          1; 1; 1; 1;         1; 1;            1;   1;"
    #      "1; 1; 1;      0; 0; 0;    0; 0; 0; 0;   0;   1; 1; 1;        1; 1; 1;    1;       1; 1;        0; 0;      0; 0; 0;        1; 1; 1;       0; 0; 0;      0; 0; 0;   1; 1;   1; 1;   1; 1; 1;    1; 1; 1; 1;      0; 0; 0; 0;     1; 1; 1; 1; 1; 1; 1;        0; 0; 0; 0; 0;     1; 1; 1;    0;   1; 1; 1;    1; 1; 1; 1; 1;        0; 0;       1; 1; 1;       1;      0; 0; 0;    1;     0; 0; 0; 0; 0; 0;          1; 1; 1; 1;         1; 1;            1;   1;"
    #      "1; 1; 1;      0; 0; 0;    0; 0; 0; 0;   0;   1; 1; 1;        1; 1; 1;    1;       1; 1;        0; 0;      0; 0; 0;        1; 1; 1;       0; 0; 0;      0; 0; 0;   1; 1;   1; 1;   1; 1; 1;    1; 1; 1; 1;      0; 0; 0; 0;     1; 1; 1; 1; 1; 1; 1;        0; 0; 0; 0; 0;     1; 1; 1;    0;   1; 1; 1;    1; 1; 1; 1; 1;        0; 0;       1; 1; 1;       1;      0; 0; 0;    1;     0; 0; 0; 0; 0; 0;          1; 1; 1; 1;         1; 1;            1;   1 "),
    #     dtype=float)
    #
    #
    #         # (03, 04, 06, 17, 33, 44)
    # valid_survival_labels = np.mat(
    #     ("  1; 1; 1;   0; 0; 0; 0;      1; 1; 1;    2; 2; 2;     0; 0;    2;"
    #      "  1; 1; 1;   0; 0; 0; 0;      1; 1; 1;    2; 2; 2;     0; 0;    2;"
    #      "  1; 1; 1;   0; 0; 0; 0;      1; 1; 1;    2; 2; 2;     0; 0;    2;"
    #      "  1; 1; 1;   0; 0; 0; 0;      1; 1; 1;    2; 2; 2;     0; 0;    2"),dtype=float)
    #
    # valid_grad_labels = np.mat(
    #     ("  1; 1; 1;   1; 1; 1; 1;     1; 1; 1;     0; 0; 0;      0; 0;    0;"
    #      "  1; 1; 1;   1; 1; 1; 1;     1; 1; 1;     0; 0; 0;      0; 0;    0;"
    #      "  1; 1; 1;   1; 1; 1; 1;     1; 1; 1;     0; 0; 0;      0; 0;    0;"
    #       " 1; 1; 1;   1; 1; 1; 1;     1; 1; 1;     0; 0; 0;      0; 0;    0"),dtype=float)
# ===================================================================================================== 3_data_hg
    #2, 3, 10, 20, 21, 22, 23, 27, 30, 35, 40
    train_survival_labels = np.mat((    " 0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 0; 0;           1; 1;  0; 0; 0;     0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1;"
                                         "0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 0; 0;           1; 1;  0; 0; 0;     0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1;"
                                         "0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 0; 0;           1; 1;  0; 0; 0;     0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1;"
                                         "0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 0; 0;           1; 1;  0; 0; 0;     0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1"),
                                         dtype=float)

    train_grad_labels = np.mat((
                                        "1; 1; 1;       0; 0; 0;    0; 0; 0;    0; 0; 0;    1; 1;      1; 1; 1;        1; 1; 1; 1;      1; 1; 1;   1; 1; 1;    1;   1; 1;"
                                        "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
                                        "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
                                        "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1"),
                                        dtype=float)
    # 4, 13, 17, 25, 46
    valid_survival_labels = np.mat((
                                        " 0; 0; 0; 0; 0; 0;  1; 1; 1; 0; 0; 0; 0; 0; 1; 1; "
                                        " 0; 0; 0; 0; 0; 0;  1; 1; 1; 0; 0; 0; 0; 0; 1; 1; "
                                        " 0; 0; 0; 0; 0; 0;  1; 1; 1; 0; 0; 0; 0; 0; 1; 1; "
                                        " 0; 0; 0; 0; 0; 0;  1; 1; 1; 0; 0; 0; 0; 0; 1; 1"),
                                        dtype=float)

    valid_grad_labels = np.mat((
                                       " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0;"
                                       " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0;"
                                       " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0;"
                                       " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0"),
                                   dtype=float)
    # 11, 16, 31, 34
    test_survival_labels = np.mat((
                                "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1;  "
                                "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1;  "
                                "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1;  "
                                "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1 "),
                                dtype=float)

    test_grad_labels = np.mat((
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;"
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; "
                                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0"),
                                  dtype=float)
    # ========================

    train_survival_labels_cat = (tf.keras.utils.to_categorical(train_survival_labels))
    train_grad_labels_cat = (tf.keras.utils.to_categorical(train_grad_labels))
    test_survival_labels_cat = (tf.keras.utils.to_categorical(test_survival_labels))
    test_grad_labels_cat = (tf.keras.utils.to_categorical(test_grad_labels))
    valid_survival_labels_cat = (tf.keras.utils.to_categorical(valid_survival_labels))
    valid_grad_labels_cat = (tf.keras.utils.to_categorical(valid_grad_labels))

# ================================================================================== Concatenate
    Concat_Train_T1 = np.concatenate((Resized_Train_T1, Rot_90_Train_T1, Rot_lr_Train_T1, Rot_ud_Train_T1), axis=0)
    # Concat_Train_CBV = np.concatenate((Resized_Train_CBV, Rot_90_Train_CBV, Rot_lr_Train_CBV, Rot_ud_Train_CBV), axis=0)
    # Concat_Train_MD = np.concatenate((Resized_Train_MD, Rot_90_Train_MD, Rot_lr_Train_MD, Rot_ud_Train_MD), axis=0)

#===================================normalization
    # max_T1 = Concat_Train_T1.max()
    # max_CBV = Concat_Train_CBV.max()
    # max_MD = Concat_Train_MD.max()
    # Concat_Train_T1_norm = Concat_Train_T1.astype('float32') / max_T1
    # Concat_Train_CBV_norm = Concat_Train_CBV.astype('float32') / max_CBV
    # Concat_Train_MD_nrom = Concat_Train_MD.astype('float32') / max_MD

    # train_dataset = np.stack((Concat_Train_T1_norm, Concat_Train_CBV_norm, Concat_Train_MD_nrom), axis=3)  # axis=0:channel first, axis=1:channel last
    # train_dataset = Concat_Train_T1
    train_T1_norm = tf.keras.utils.normalize(Concat_Train_T1, axis=1)
    train_dataset =  train_T1_norm.reshape(120, 5, 64, 64, 1)

#===================validation
    Concat_Valid_T1 = np.concatenate((Resized_Valid_T1, Rot_90_Valid_T1, Rot_lr_Valid_T1, Rot_ud_Valid_T1), axis=0)
    # Concat_Valid_CBV = np.concatenate((Resized_Valid_CBV, Rot_90_Valid_CBV, Rot_lr_Valid_CBV, Rot_ud_Valid_CBV), axis=0)
    # Concat_Valid_MD = np.concatenate((Resized_Valid_MD, Rot_90_Valid_MD, Rot_lr_Valid_MD, Rot_ud_Valid_MD), axis=0)

    # max_T1_valid = Concat_Valid_T1.max()
    # max_CBV_valid = Concat_Valid_CBV.max()
    # max_MD_valid = Concat_Valid_MD.max()
    # Concat_Valid_T1_norm = Concat_Valid_T1.astype('float32') / max_T1_valid
    # Concat_Valid_CBV_norm = Concat_Valid_CBV.astype('float32') / max_CBV_valid
    # Concat_Valid_MD_norm = Concat_Valid_MD.astype('float32') / max_MD_valid

    # valid_dataset = np.stack((Concat_Valid_T1_norm, Concat_Valid_MD_norm, Concat_Valid_CBV_norm), axis=3)  # axis=0:channel first, axis=1:channel last
    valid_T1_norm = tf.keras.utils.normalize(Concat_Valid_T1, axis=1)
    valid_dataset = valid_T1_norm.reshape(64, 5, 64, 64, 1)

#============== test
    Concat_Test_T1 = np.concatenate((Resized_Test_T1, Rot_90_Test_T1, Rot_lr_Test_T1, Rot_ud_Test_T1), axis=0)
    # Concat_Test_CBV = np.concatenate((Resized_Test_CBV, Rot_90_Test_CBV, Rot_lr_Test_CBV, Rot_ud_Test_CBV), axis=0)
    # Concat_Test_MD = np.concatenate((Resized_Test_MD, Rot_90_Test_MD, Rot_lr_Test_MD, Rot_ud_Test_MD), axis=0)
    #
    # max_T1_test = Concat_Test_T1.max()
    # max_CBV_test = Concat_Test_CBV.max()
    # max_MD_test = Concat_Test_MD.max()
    #
    # Concat_Test_T1_norm = Concat_Test_T1.astype('float32') / max_T1_test
    # Concat_Test_CBV_norm = Concat_Test_CBV.astype('float32') / max_CBV_test
    # Concat_Test_MD_norm = Concat_Test_MD.astype('float32') / max_MD_test
    #
    # test_dataset = np.stack((Concat_Test_T1_norm, Concat_Test_CBV_norm, Concat_Test_MD_norm), axis=-1)
    test_T1_norm = tf.keras.utils.normalize(Concat_Test_T1, axis=1)
    test_dataset = test_T1_norm.reshape(56, 5, 64, 64, 1)
 #==========================================================================================================shuffle data
    # def shuffle_lists(t1, md, cbv, y_survival_cat, y_grad_cat):
    #     index_shuf = list(range(1400))
    #     shuffle(index_shuf)
    #     t1_sn = np.hstack([t1[sn]]
    #                       for sn in index_shuf)
    #     md_sn = np.hstack([md[sn]]
    #                       for sn in index_shuf)
    #     cbv_sn = np.hstack([cbv[sn]]
    #                       for sn in index_shuf)
    #     y_survival_cat_sn = np.hstack([y_survival_cat[sn]]
    #                       for sn in index_shuf)
    #     y_grad_cat_sn = np.hstack([y_grad_cat[sn]]
    #                       for sn in index_shuf)
    #     return t1_sn, md_sn, cbv_sn, y_survival_cat_sn, y_grad_cat_sn
    #
    # t1_sn, cbv_sn, md_sn, y_survival_cat_sn, y_grad_cat_sn = shuffle_lists(Concat_Train_T1_norm , Concat_Train_CBV_norm, Concat_Train_MD_norm, train_survival_labels_cat, train_grad_labels_cat)
    #
    # t1_sn = t1_sn.reshape(len(Concat_Train_T1), 64, 64)
    # cbv_sn = cbv_sn.reshape(len(Concat_Train_CBV), 64, 64)
    # md_sn = md_sn.reshape(len(Concat_Train_MD), 64, 64)
    #
    # y_survival_cat_sn = y_survival_cat_sn.reshape(len(train_survival_labels_cat), 3)
    # y_grad_cat_sn = y_grad_cat_sn.reshape(len(train_grad_labels_cat), 2)

    # train_dataset = np.stack((t1_sn, cbv_sn, md_sn), -1)

#==========================================================================
    # train_dataset = train_dataset[..., np.newaxis]
    #train_dataset = train_dataset.astype('float32') / 255.
    # train_dataset = tf.keras.utils.normalize(train_dataset, axis=1)

#==================================================================================
    logno =35
    length = 64
    channel = 1
    volume = 5
    batch_size = 10
    learning_rate = 0.000001
    n_output_survival = 2
    n_output_grad = 2
    total_size = train_dataset.shape[0]

#================================================================
    input = tf.placeholder(tf.float32, [None, volume, length, length, channel], name='input')
    labels_grad = tf.placeholder(tf.float32, [None, n_output_grad], name='grad_labels')
    labels_survival = tf.placeholder(tf.float32, [None, n_output_survival], name='survival_labels')
    is_training = tf.placeholder(tf.bool, name='is_training')
    # global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
#====================================================================================================
    output_graph = True
    net = Network(
        n_output_grad=n_output_grad,
        n_output_survival=n_output_survival,
        n_length=length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        channel=channel,
        output_graph=output_graph,
        use_ckpt=False,
        logno=logno,

    )
# =================================================================================
    output_grad, output_survival = net._build_net(input, is_training)

    accuracy_grad = accuracy(output_grad, labels_grad)
    accuracy_survival = accuracy(output_survival, labels_survival)

    tf.summary.scalar("accuracy_grade", accuracy_grad)
    tf.summary.scalar("accuracy_survival", accuracy_survival)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    dir_path = '/exports/lkeb-hpc/gkarami/Code/Log/' + str(logno)  # os.path.dirname(os.path.realpath(__file__))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    epoch = 15000


    with tf.variable_scope('loss'):
        cross_entropy = -tf.reduce_mean(labels_grad * tf.log(output_grad))\
                        -tf.reduce_mean(labels_survival * tf.log(output_survival))\
                        # + tf.reduce_mean((labels_grad + output_grad) - 2 * (labels_grad * output_grad ))\
                        # + tf.reduce_mean((labels_survival + output_survival) - 2 * (labels_survival * output_survival))

        # +int(labels_grad) + int(output_grad) - 2* labels_grad * output_grad \
                        # +labels_survival + output_survival - 2* labels_survival * output_survival

    tf.summary.scalar("cross_entropy", cross_entropy)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    with tf.control_dependencies(extra_update_ops):
        # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)
        train_op = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(cross_entropy)#, global_step=global_step)
    summ = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    iteration = int(total_size / batch_size)

    i = 1  # total training time
    accu_train_grad = []
    accu_valid_grad = []
    accu_test_grad = []
    accu_train_survival = []
    accu_valid_survival = []
    accu_test_survival = []
    loss_train = []
    loss_valid = []
    loss_test = []

    train_writer = tf.summary.FileWriter((dir_path) + '/train/', graph=tf.get_default_graph())
    validation_writer = tf.summary.FileWriter((dir_path) + '/validation/', graph=sess.graph)
    test_writer = tf.summary.FileWriter((dir_path) + '/test', graph=sess.graph)

    point = 0
    for e in range(epoch):
        print("-------------------------------")
        print("epoch %d" % (e + 1))
        # randomly sample batch memory from all memory
        indices = np.random.permutation(total_size)

        for ite in range(iteration):
            point = point + 1
            mini_indices = indices[ite * batch_size:(ite + 1) * batch_size]
            batch_x = train_dataset[mini_indices, :, :, :]
            batch_y_grad = train_grad_labels_cat[mini_indices, :]
            batch_y_survival = train_survival_labels_cat[mini_indices, :]

            # train eval network
            _, cost, sum_train, pred_grad, pred_survival, acc_g, acc_s = sess.run([train_op, cross_entropy,
                                                                                   summ, output_grad, output_survival,
                                                                                   accuracy_grad, accuracy_survival],

                                                                                  feed_dict={input: batch_x,
                                                                                             labels_grad: batch_y_grad,
                                                                                             labels_survival: batch_y_survival,
                                                                                             is_training: True,

                                                                                             })

            print('******Train, step: %d , loss: %f, ,acc_g :%f, acc_s:%f *******' % (point, cost, acc_g, acc_s))
            print(pred_grad, batch_y_grad)
            print(pred_survival, batch_y_survival)

            train_writer.add_summary(sum_train, point)

            cost_vl, rs, pred_grad, pred_survival, acc_g, acc_s = sess.run(
                [cross_entropy, summ, output_grad, output_survival, accuracy_grad, accuracy_survival],

                feed_dict={
                    labels_grad: valid_grad_labels_cat,
                    labels_survival: valid_survival_labels_cat,
                    input: valid_dataset, is_training: False,

                })

            if i % 20 == 0:
                train_writer.flush()
                validation_writer.flush()

            print('******Validation, step: %d , loss: %f,acc_g :%f, acc_s:%f*******' % (point, cost_vl, acc_g, acc_s))
            # print(pred_grad, valid_grad_labels_cat)
            # print(pred_survival, valid_survival_labels_cat)

            validation_writer.add_summary(rs, point)

            cost_vl, rs, pred_grad, pred_survival, acc_g, acc_s = sess.run(
                [cross_entropy, summ, output_grad, output_survival, accuracy_grad, accuracy_survival],

                feed_dict={
                    labels_grad: test_grad_labels_cat,
                    labels_survival: test_survival_labels_cat,
                    input: test_dataset, is_training: False,
                })


            if i % 20 == 0:
                train_writer.flush()
                test_writer.flush()
            print('******Test, step: %d , loss: %f, acc_g:%f*******' % (point, cost_vl, acc_g))
            # print(pred_grad, test_grad_labels_cat)
            # print(pred_survival, test_survival_labels_cat)

            test_writer.add_summary(rs, point)

            if i % 50 == 0:
                # saver = tf.train.Saver()
                if not os.path.exists(dir_path + "/weights_saved"):
                    os.mkdir(dir_path + "/weights_saved")
                saver_path = saver.save(sess,
                                        dir_path + '/weights_saved/model.ckpt')  # save model into save/model.ckpt file
                print('Model saved in file:', saver_path)

            # if i%5==0: # save histogram
            #     net.merge_hist(batch_x,batch_y_grad,batch_y_survival,point)
            # i = i+1

        # early stopping
        # if train_rate_grad==100 and train_rate_survival==100:
        #     if early_stop==10:
        #         print("Early Stopping!")
        #         break
        #     else:early_stop = early_stop+1

    # net.plot_cost() # plot trainingi cost

    plt.figure()  # plot loss
    plt.plot(np.arange(len(loss_train)), loss_train, label='train', linestyle='--')
    plt.plot(np.arange(len(loss_valid)), loss_valid, label='valid', linestyle='-')
    plt.plot(np.arange(len(loss_test)), loss_test, label='test', linestyle=':')
    plt.ylabel('loss_ dataNorma')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("/exports/lkeb-hpc/gkarami/Code/Jobs/" + str(logno) + "_loss_jointly.png")
    # plt.show()

    plt.figure()  # plot accuracy
    plt.plot(np.arange(len(accu_train_grad)), accu_train_grad, label='train grad', linestyle='--')
    plt.plot(np.arange(len(accu_valid_grad)), accu_valid_grad, label='valid grad', linestyle='-')
    plt.plot(np.arange(len(accu_test_grad)), accu_test_grad, label='test grad', linestyle=':')
    plt.ylabel('grad accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("/exports/lkeb-hpc/gkarami/Code/Jobs/" + str(logno) + "_accuracy_jointly_grad.png")
    # plt.show()

    plt.figure()  # plot accuracy
    plt.plot(np.arange(len(accu_train_survival)), accu_train_survival, label='train survival', linestyle='--')
    plt.plot(np.arange(len(accu_valid_survival)), accu_valid_survival, label='valid survival', linestyle='-')
    plt.plot(np.arange(len(accu_test_survival)), accu_test_survival, label='test survival', linestyle=':')
    plt.ylabel('survival accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("/exports/lkeb-hpc/gkarami/Code/Jobs/" + str(logno) + "_accuracy_jointly_survival.png")
    # plt.show()
    # ==========================================


def main():
    train_model()


if __name__ == '__main__':
    main()

