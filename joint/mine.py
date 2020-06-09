import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import glob
import tensorflow as tf
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
##########################################
Path_Train_T1 = '/exports/lkeb-hpc/gkarami/Data/2_Train_T1/'
Path_Train_MD = '/exports/lkeb-hpc/gkarami/Data/2_Train_MD/'
Path_Train_CBV = '/exports/lkeb-hpc/gkarami/Data/2_Train_CBV/'
Path_Train_Masks = '/exports/lkeb-hpc/gkarami/Data/2_Train_T1/'

Path_Validation_T1 = '/exports/lkeb-hpc/gkarami/Data/2_Validation_T1/'
Path_Validation_MD = '/exports/lkeb-hpc/gkarami/Data/2_Validation_MD/'
Path_Validation_CBV = '/exports/lkeb-hpc/gkarami/Data/2_Validation_CBV/'
Path_Validation_Masks = '/exports/lkeb-hpc/gkarami/Data/2_Validation_T1/'

Path_Test_T1 = '/exports/lkeb-hpc/gkarami/Data/2_Test_T1/'
Path_Test_MD = '/exports/lkeb-hpc/gkarami/Data/2_Test_MD/'
Path_Test_CBV = '/exports/lkeb-hpc/gkarami/Data/2_Test_CBV/'
Path_Test_Masks = '/exports/lkeb-hpc/gkarami/Data/2_Test_T1/'

IMG_Train_T1 = sorted(next(os.walk(Path_Train_T1))[2])
IMG_Train_MD = sorted(next(os.walk(Path_Train_MD))[2])
IMG_Train_CBV = sorted(next(os.walk(Path_Train_CBV))[2])
IMG_Train_Masks = IMG_Train_T1

IMG_Validation_T1 = sorted(next(os.walk(Path_Validation_T1))[2])
IMG_Validation_MD = sorted(next(os.walk(Path_Validation_MD))[2])
IMG_Validation_CBV = sorted(next(os.walk(Path_Validation_CBV))[2])
IMG_Validation_Masks = IMG_Validation_T1

IMG_Test_T1 = sorted(next(os.walk(Path_Test_T1))[2])
IMG_Test_MD = sorted(next(os.walk(Path_Test_MD))[2])
IMG_Test_CBV = sorted(next(os.walk(Path_Test_CBV))[2])
IMG_Test_Masks = IMG_Test_T1

print(IMG_Train_T1)
print(IMG_Test_CBV)
print(IMG_Validation_T1)

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
IMG_Subjects = 2

Inputs_Train_T1 = np.zeros((len(IMG_Train_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Train_MD = np.zeros((len(IMG_Train_MD), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Train_CBV = np.zeros((len(IMG_Train_CBV), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Train_Masks = np.zeros((len(IMG_Train_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

Resized_Train_T1 = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Resized_Train_CBV = np.zeros((len(IMG_Train_CBV), 64, 64), dtype=np.float)
Resized_Train_MD = np.zeros((len(IMG_Train_MD), 64, 64), dtype=np.float)
Resized_Train_Masks = np.zeros((len(IMG_Train_T1), 64, 64, 1), dtype=np.bool)

Inputs_Validation_T1 = np.zeros((len(IMG_Validation_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Validation_MD = np.zeros((len(IMG_Validation_MD), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Validation_CBV = np.zeros((len(IMG_Validation_CBV), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Validation_Masks = np.zeros((len(IMG_Validation_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

Resized_Validation_T1 = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Resized_Validation_CBV = np.zeros((len(IMG_Validation_CBV), 64, 64), dtype=np.float)
Resized_Validation_MD = np.zeros((len(IMG_Validation_MD), 64, 64), dtype=np.float)
Resized_Validation_Masks = np.zeros((len(IMG_Validation_T1), 64, 64, 1), dtype=np.bool)

Inputs_Test_T1 = np.zeros((len(IMG_Test_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Test_MD = np.zeros((len(IMG_Test_MD), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Test_CBV = np.zeros((len(IMG_Test_CBV), IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
Inputs_Test_Masks = np.zeros((len(IMG_Test_T1), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

Resized_Test_T1 = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Resized_Test_CBV = np.zeros((len(IMG_Test_CBV), 64, 64), dtype=np.float)
Resized_Test_MD = np.zeros((len(IMG_Test_MD), 64, 64), dtype=np.float)
Resized_Test_Masks = np.zeros((len(IMG_Test_T1), 64, 64, 1), dtype=np.bool)

IMG_HEIGHT_resized, IMG_WIDTH_resized = 64, 64

# =============================================================== extract ROI first for T1 Train
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Train_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Train_T1, image_ID)
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
                                 (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                 mode='constant',
                                 anti_aliasing=True,
                                 preserve_range=True)

    Resized_Train_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                   (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                   mode='constant',
                                                   anti_aliasing=True,
                                                   preserve_range=True), axis=-1)

    Inputs_Train_T1[n] = image
    Inputs_Train_Masks[n] = mask

    n += 1

# # ===================================================================== CBV  Train
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Train_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Train_CBV, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Train_CBV[n] = resize(cropped_image[:, :],
                                  (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                  mode='constant',
                                  anti_aliasing=True,
                                  preserve_range=True)

    Resized_Train_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                   (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                   mode='constant',
                                                   anti_aliasing=True,
                                                   preserve_range=True), axis=-1)

    Inputs_Train_CBV[n] = image
    Inputs_Train_Masks[n] = mask

    n += 1

#  ============================================================ MD  Train
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Train_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Train_MD, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Train_MD[n] = resize(cropped_image[:, :],
                                 (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                 mode='constant',
                                 anti_aliasing=True,
                                 preserve_range=True)

    Resized_Train_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                   (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                   mode='constant',
                                                   anti_aliasing=True,
                                                   preserve_range=True), axis=-1)

    Inputs_Train_MD[n] = image
    Inputs_Train_Masks[n] = mask

    n += 1

# ================================================== for  T1   Test
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Test_T1)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Test_T1, image_ID)
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
                                (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                mode='constant',
                                anti_aliasing=True,
                                preserve_range=True)

    Resized_Test_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                  (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                  mode='constant',
                                                  anti_aliasing=True,
                                                  preserve_range=True), axis=-1)

    Inputs_Test_T1[n] = image
    Inputs_Test_Masks[n] = mask

    n += 1

# ============================================================ for  CBV  Test
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Test_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Test_CBV, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Test_CBV[n] = resize(cropped_image[:, :],
                                 (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                 mode='constant',
                                 anti_aliasing=True,
                                 preserve_range=True)

    Resized_Test_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                  (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                  mode='constant',
                                                  anti_aliasing=True,
                                                  preserve_range=True), axis=-1)

    Inputs_Test_CBV[n] = image
    Inputs_Test_Masks[n] = mask

    n += 1

# # # =========================================================for MD Test
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Test_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Test_MD, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Test_MD[n] = resize(cropped_image[:, :],
                                (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                mode='constant',
                                anti_aliasing=True,
                                preserve_range=True)

    Resized_Test_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                  (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                  mode='constant',
                                                  anti_aliasing=True,
                                                  preserve_range=True), axis=-1)

    Inputs_Test_MD[n] = image
    Inputs_Test_Masks[n] = mask

    n += 1

# ===================================================for  T1  validation
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Validation_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Validation_T1, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Validation_T1[n] = resize(cropped_image[:, :],
                                      (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                      mode='constant',
                                      anti_aliasing=True,
                                      preserve_range=True)

    Resized_Validation_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                        (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                        mode='constant',
                                                        anti_aliasing=True,
                                                        preserve_range=True), axis=-1)

    Inputs_Validation_T1[n] = image
    Inputs_Validation_Masks[n] = mask

    n += 1

# # ====================================================================== for CBV  validation
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Validation_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Validation_CBV, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Validation_CBV[n] = resize(cropped_image[:, :],
                                       (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                       mode='constant',
                                       anti_aliasing=True,
                                       preserve_range=True)

    Resized_Validation_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                        (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                        mode='constant',
                                                        anti_aliasing=True,
                                                        preserve_range=True), axis=-1)

    Inputs_Validation_CBV[n] = image
    Inputs_Validation_Masks[n] = mask

    n += 1

# =======================================for validation MD
n = 0
for mask_path in glob.glob('{}/*.tif'.format(Path_Validation_Masks)):
    base = os.path.basename(mask_path)
    image_ID, ext = os.path.splitext(base)
    image_path = '{}/{}.tif'.format(Path_Validation_MD, image_ID)
    mask = imread(mask_path)
    image = imread(image_path)

    y_coord, x_coord = np.where(mask != 0)

    y_min = min(y_coord)
    y_max = max(y_coord)
    x_min = min(x_coord)
    x_max = max(x_coord)

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    Resized_Validation_MD[n] = resize(cropped_image[:, :],
                                      (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                      mode='constant',
                                      anti_aliasing=True,
                                      preserve_range=True)

    Resized_Validation_Masks[n] = np.expand_dims(resize(cropped_mask,
                                                        (IMG_HEIGHT_resized, IMG_WIDTH_resized),
                                                        mode='constant',
                                                        anti_aliasing=True,
                                                        preserve_range=True), axis=-1)

    Inputs_Validation_MD[n] = image
    Inputs_Validation_Masks[n] = mask

    n += 1
# =====================================================  showthe images
# rows = 3
# columns = 4
# Figure = plt.figure(figsize=(15,15))
# Image_List = [Inputs_Train_T1[0], Resized_Train_T1[0], Resized_Train_MD[0], Resized_Train_CBV[0],Inputs_Validation_T1[0], Resized_Validation_T1[0], Resized_Validation_MD[0], Resized_Validation_CBV[0]
#               , Inputs_Test_T1[0], Resized_Test_T1[0], Resized_Test_MD[0], Resized_Test_CBV[0]]
#
# for i in range(1, rows*columns + 1):
#     Image = Image_List[i-1]
#     Sub_Plot_Image = Figure.add_subplot(rows, columns, i)
#     Sub_Plot_Image.imshow(np.squeeze(Image))
# plt.show()

# imshow(Resized_Train_MD[5])
# plt.show()
# ==================================  Augmentation
Rot_90_Train_T1 = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Rot_lr_Train_T1 = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Rot_ud_Train_T1 = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Train_T1)):
    img = Resized_Train_T1[m]
    Rot_90_Train_T1[m] = np.rot90(img)
    Rot_lr_Train_T1[m] = np.fliplr(img)
    Rot_ud_Train_T1[m] = np.flipud(img)

Rot_90_Train_CBV = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Rot_lr_Train_CBV = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Rot_ud_Train_CBV = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Train_T1)):
    img = Resized_Train_CBV[m]
    Rot_90_Train_CBV[m] = np.rot90(img)
    Rot_lr_Train_CBV[m] = np.fliplr(img)
    Rot_ud_Train_CBV[m] = np.flipud(img)

Rot_90_Train_MD = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Rot_lr_Train_MD = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
Rot_ud_Train_MD = np.zeros((len(IMG_Train_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Train_T1)):
    img = Resized_Train_MD[m]
    Rot_90_Train_MD[m] = np.rot90(img)
    Rot_lr_Train_MD[m] = np.fliplr(img)
    Rot_ud_Train_MD[m] = np.flipud(img)
####
Rot_90_Validation_T1 = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Rot_lr_Validation_T1 = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Rot_ud_Validation_T1 = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Validation_T1)):
    img = Resized_Validation_T1[m]
    Rot_90_Validation_T1[m] = np.rot90(img)
    Rot_lr_Validation_T1[m] = np.fliplr(img)
    Rot_ud_Validation_T1[m] = np.flipud(img)

Rot_90_Validation_CBV = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Rot_lr_Validation_CBV = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Rot_ud_Validation_CBV = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Validation_T1)):
    img = Resized_Validation_CBV[m]
    Rot_90_Validation_CBV[m] = np.rot90(img)
    Rot_lr_Validation_CBV[m] = np.fliplr(img)
    Rot_ud_Validation_CBV[m] = np.flipud(img)

Rot_90_Validation_MD = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Rot_lr_Validation_MD = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
Rot_ud_Validation_MD = np.zeros((len(IMG_Validation_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Validation_T1)):
    img = Resized_Validation_MD[m]
    Rot_90_Validation_MD[m] = np.rot90(img)
    Rot_lr_Validation_MD[m] = np.fliplr(img)
    Rot_ud_Validation_MD[m] = np.flipud(img)
####
Rot_90_Test_T1 = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Rot_lr_Test_T1 = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Rot_ud_Test_T1 = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Test_T1)):
    img = Resized_Test_T1[m]
    Rot_90_Test_T1[m] = np.rot90(img)
    Rot_lr_Test_T1[m] = np.fliplr(img)
    Rot_ud_Test_T1[m] = np.flipud(img)

Rot_90_Test_CBV = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Rot_lr_Test_CBV = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Rot_ud_Test_CBV = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Test_T1)):
    img = Resized_Test_CBV[m]
    Rot_90_Test_CBV[m] = np.rot90(img)
    Rot_lr_Test_CBV[m] = np.fliplr(img)
    Rot_ud_Test_CBV[m] = np.flipud(img)

Rot_90_Test_MD = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Rot_lr_Test_MD = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
Rot_ud_Test_MD = np.zeros((len(IMG_Test_T1), 64, 64), dtype=np.float)
for m in range(len(IMG_Test_T1)):
    img = Resized_Test_MD[m]
    Rot_90_Test_MD[m] = np.rot90(img)
    Rot_lr_Test_MD[m] = np.fliplr(img)
    Rot_ud_Test_MD[m] = np.flipud(img)
    # ================================================== show the augmentation
    # rows = 9
    # columns = 4
    # Figure = plt.figure(figsize=(15,15))
    # Image_List = [Resized_Train_T1[34], Rot_90_Train_T1[34], Rot_lr_Train_T1[34], Rot_ud_Train_T1[34],
    #               Resized_Train_MD[34], Rot_90_Train_MD[34], Rot_lr_Train_MD[34], Rot_ud_Train_MD[34],
    #               Resized_Train_CBV[34], Rot_90_Train_CBV[34], Rot_lr_Train_CBV[34], Rot_ud_Train_CBV[34],
    #
    #               Resized_Test_T1[22], Rot_90_Test_T1[22], Rot_lr_Test_T1[22], Rot_ud_Test_T1[22],
    #               Resized_Test_MD[22], Rot_90_Test_MD[22], Rot_lr_Test_MD[22], Rot_ud_Test_MD[22],
    #               Resized_Test_CBV[22], Rot_90_Test_CBV[22], Rot_lr_Test_CBV[22], Rot_ud_Test_CBV[22],
    #
    #               Resized_Validation_T1[35], Rot_90_Validation_T1[35], Rot_lr_Validation_T1[35], Rot_ud_Validation_T1[35],
    #               Resized_Validation_MD[35], Rot_90_Validation_MD[35], Rot_lr_Validation_CBV[35], Rot_ud_Validation_CBV[35],
    #               Resized_Validation_CBV[35], Rot_90_Validation_CBV[35], Rot_lr_Validation_CBV[35], Rot_ud_Validation_CBV[35]]
    #
    # for i in range(1, rows*columns + 1):
    #     Image = Image_List[i-1]
    #     Sub_Plot_Image = Figure.add_subplot(rows, columns, i)
    #     Sub_Plot_Image.imshow(np.squeeze(Image))
    # plt.show()

    # ========================= Data/2_Train_T1 data_2_train_T1

train_survival_labels = np.mat(("0; 0; 0; 0; 1; 1; 1; 2; 2; 2; 1; 1; 1; 2; 2; 0; 0; 0; 2; 2; 2; 1; 1; 1; 0; 0; 0; 0; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 2; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2;"
                                    "0; 0; 0; 0; 1; 1; 1; 2; 2; 2; 1; 1; 1; 2; 2; 0; 0; 0; 2; 2; 2; 1; 1; 1; 0; 0; 0; 0; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 2; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2;"
                                    "0; 0; 0; 0; 1; 1; 1; 2; 2; 2; 1; 1; 1; 2; 2; 0; 0; 0; 2; 2; 2; 1; 1; 1; 0; 0; 0; 0; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 2; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2;"
                                    "0; 0; 0; 0; 1; 1; 1; 2; 2; 2; 1; 1; 1; 2; 2; 0; 0; 0; 2; 2; 2; 1; 1; 1; 0; 0; 0; 0; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 2; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2" ), dtype=float)

train_grad_labels = np.mat((       "1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0; 0; 1; 0; 0; 0; 0; 0;"
                                    "1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0; 0; 1; 0; 0; 0; 0; 0;"
                                    "1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0; 0; 1; 0; 0; 0; 0; 0;"
                                    "1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0; 0; 1; 0; 0; 0; 0; 0"),  dtype=float)

test_survival_labels = np.mat((" 1; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 0; 1; 1; 2; 2; 1; 1; 2; 2;   1; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 0; 1; 1; 2; 2; 1; 1; 2; 2;   1; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 0; 1; 1; 2; 2; 1; 1; 2; 2;     1; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 0; 1; 1; 2; 2; 1; 1; 2; 2  "), dtype=float)
test_grad_labels = np.mat(( " 1; 1; 1;  0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0;   1; 1; 1;  0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0;    1; 1; 1;  0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0;    1; 1; 1;  0; 0; 0; 1; 1; 1; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 0"), dtype=float)

validation_survival_labels = np.mat((" 1; 1; 1; 1; 1; 1; 0; 0; 0; 2; 2; 0; 0; 1; 1; 1; 1; 2;          1; 1; 1; 1; 1; 1; 0; 0; 0; 2; 2; 0; 0; 1; 1; 1; 1; 2;         1; 1; 1; 1; 1; 1; 0; 0; 0; 2; 2; 0; 0; 1; 1; 1; 1; 2;           1; 1; 1; 1; 1; 1; 0; 0; 0; 2; 2; 0; 0; 1; 1; 1; 1; 2  "), dtype=float)
validation_grad_labels = np.mat(( " 1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0;  1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0;   1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0;   1; 1; 1; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0"), dtype=float)
#######################################################
    # y_train = np.mat(("0; 0; 0; 0; 1; 1; 1; 2; 2; 2; 1; 1; 1; 2; 2; 0; 0; 0; 2; 2; 2; 1; 1; 1; 0; 0; 0; 0; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 2; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2"), dtype=float)
    # y_test = np.mat((" 1; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 0; 1; 1; 2; 2; 1; 1; 2; 2  "), dtype=float)
    # y_validation = np.mat((" 1; 1; 1; 1; 1; 1; 0; 0; 0; 2; 2; 0; 0; 1; 1; 1; 1; 2 "), dtype=float)


    # y_train = np.mat((
    #               "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
    #               "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
    #               "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; "
    #               "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2 "),
    #               dtype=float)
    #
    # y_validation = np.mat((
    #                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; "
    #                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; "
    #                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; "
    #                " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2  "),
    #                 dtype=float)
    #
    # y_test = np.mat((
    #                "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
    #                "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
    #                "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;"
    #                "1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2"),
    #                 dtype=float)

    #======================
    # train_survival_labels = np.mat(( " 1; 1; 1; 1; 1; 2; 2; 2; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1;"
    #                                  "1; 1; 1; 1; 1; 2; 2; 2; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1;"
    #                                  "1; 1; 1; 1; 1; 2; 2; 2; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1;"
    #                                  "1; 1; 1; 1; 1; 2; 2; 2; 0; 0; 0; 2; 2; 2; 2; 2; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 1; 1,"),
    #                                  dtype=float)
    #
    # train_grad_labels = np.mat((
    #                                 "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
    #                                 "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
    #                                 "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;"
    #                                  "1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1"),
    #                                 dtype=float)
    #
    # validation_survival_labels = np.mat((
    #                                    " 0; 0; 0; 0; 0; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 2; 2;"
    #                                    " 0; 0; 0; 0; 0; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 2; 2;"
    #                                    " 0; 0; 0; 0; 0; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 2; 2;"
    #                                    " 0; 0; 0; 0; 0; 1; 1; 2; 2; 2; 0; 0; 0; 0; 0; 2; 2"),
    #                                    dtype=float)
    #
    # validation_grad_labels = np.mat((
    #                                " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0;"
    #                                " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0;"
    #                                " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0;"
    #                                " 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0"),
    #                            dtype=float)
    #
    # test_survival_labels = np.mat((
    #                                   "0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 2; 2; 2; "
    #                                   "0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 2; 2; 2; "
    #                                   "0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 2; 2; 2; "
    #                                   "0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 2; 2; 2 "),
    #                                   dtype=float)
    #
    # test_grad_labels = np.mat((
    #                               " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;"
    #                               " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0;"
    #                               " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; "
    #                               " 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0"),
    #                           dtype=float)
    # ================================Convert class vectors to binary class matrices
train_survival_labels_cat = (tf.keras.utils.to_categorical(train_survival_labels))
test_survival_labels_cat = (tf.keras.utils.to_categorical(test_survival_labels))
validation_survival_labels_cat = (tf.keras.utils.to_categorical(validation_survival_labels))

train_grad_labels_cat = (tf.keras.utils.to_categorical(train_grad_labels))
test_grad_lanbels_cat = (tf.keras.utils.to_categorical(test_grad_labels))
validation_grad_labels_cat = (tf.keras.utils.to_categorical(validation_grad_labels))

num_classes_survival = train_survival_labels_cat.shape[1]
num_classes_grad = train_grad_labels_cat.shape[1]
class_names = ['Short_Term_Survival','Medium_term-survival', 'Long_Term_Survival']


# ============================================ Concatenat each series
Concat_Train_T1 = np.concatenate((Resized_Train_T1, Rot_90_Train_T1, Rot_lr_Train_T1, Rot_ud_Train_T1), axis=0)
Concat_Train_CBV = np.concatenate((Resized_Train_CBV, Rot_90_Train_CBV, Rot_lr_Train_CBV, Rot_ud_Train_CBV), axis=0)
Concat_Train_MD = np.concatenate((Resized_Train_MD, Rot_90_Train_MD, Rot_lr_Train_MD, Rot_ud_Train_MD), axis=0)

Concat_Test_T1 = np.concatenate((Resized_Test_T1, Rot_90_Test_T1, Rot_lr_Test_T1, Rot_ud_Test_T1), axis=0)
Concat_Test_CBV = np.concatenate((Resized_Test_CBV, Rot_90_Test_CBV, Rot_lr_Test_CBV, Rot_ud_Test_CBV), axis=0)
Concat_Test_MD = np.concatenate((Resized_Test_MD, Rot_90_Test_MD, Rot_lr_Test_MD, Rot_ud_Test_MD), axis=0)

Concat_Validation_T1 = np.concatenate((Resized_Validation_T1, Rot_90_Validation_T1, Rot_lr_Validation_T1, Rot_ud_Validation_T1), axis=0)
Concat_Validation_CBV = np.concatenate((Resized_Validation_CBV, Rot_90_Validation_CBV, Rot_lr_Validation_CBV, Rot_ud_Validation_CBV), axis=0)
Concat_Validation_MD = np.concatenate((Resized_Validation_MD, Rot_90_Validation_MD, Rot_lr_Validation_MD, Rot_ud_Validation_MD), axis=0)

# ======================================== concatenat each group
train_data = np.stack((Concat_Train_T1, Concat_Train_MD, Concat_Train_CBV), axis=3)
validation_data = np.stack((Concat_Validation_T1, Concat_Validation_MD, Concat_Validation_CBV), axis=3) # axis=0:channel first, axis=1:channel last
test_data = np.stack((Concat_Test_T1, Concat_Test_MD, Concat_Test_CBV), axis=3)

################################################
    # imshow(train_data[1, :, :, 1])
    # plt.show()

    # rows = 3
    # columns = 3
    # Figure = plt.figure(figsize=(15,15))
    # Image_List = [train_data[1, :, :, 1], train_data[1, :, :, 2], train_data[1, :, :, 0],
    #              test_data[1, :, :, 1], test_data[1, :, :, 2], test_data[1, :, :, 0],
    #              validation_data[1, :, :, 1], validation_data[1, :, :, 2], validation_data[1, :, :, 0]]
    #
    # for i in range(1, rows*columns + 1):
    #     Image = Image_List[i-1]
    #     Sub_Plot_Image = Figure.add_subplot(rows, columns, i)
    #     Sub_Plot_Image.imshow(np.squeeze(Image))
    # plt.show()
    # =========================================== shuffle trrain data
    # def shuffle_lists( T1, MD, CBV, y_cat):
    #     index_shuf = list(range(1400))
    #     shuffle(index_shuf)
    #     T1_sn = np.hstack([T1[sn]]
    #                              for sn in index_shuf)
    #     MD_sn = np.hstack([MD[sn]]
    #                              for sn in index_shuf)
    #     CBV_sn = np.hstack([CBV[sn]]
    #                               for sn in index_shuf)
    #     y_cat_sn = np.hstack([y_cat[sn]]
    #                               for sn in index_shuf)
    #     return T1_sn, MD_sn, CBV_sn, y_cat_sn
    #
    # # if __name__=='__main__':
    #
    #
    # T1_sn, MD_sn, CBV_sn, y_cat_sn = shuffle_lists( Concat_Train_T1, Concat_Train_MD, Concat_Train_CBV, y_train_cat)
    # T1_sn = T1_sn.reshape(1400, 64, 64)
    # MD_sn = MD_sn.reshape(1400, 64, 64)
    # CBV_sn = CBV_sn.reshape(1400, 64, 64)
    # y_cat_sn = y_cat_sn.reshape(1400, 3)
    #
    # train_data = np.stack((T1_sn, MD_sn, CBV_sn), axis=3)  # axis=0:channel first, axis=1:channel last
    # y_train_cat = y_cat_sn

#========================================================= reshape  for conv3d
train_reshape  = train_data.reshape(280, 5, 64, 64, 3)
validation_reshape  = validation_data.reshape(72, 5, 64, 64, 3)
test_reshape  = test_data.reshape(80, 5, 64, 64, 3)

    # train_reshape = Inputs_Train_CBV.reshape(280, 5, 64, 64, 1)
    # test_reshape=Inputs_Test_CBV.reshape(80, 5, 64, 64, 1)
    # validation_reshape=Inputs_Validation_CBV.reshape(72, 5, 64, 64, 1)
    # input_shape = (5, 64, 64, 1)

 # ==================================== normalization
train_norm = tf.keras.utils.normalize(train_reshape, axis=1)
test_norm = tf.keras.utils.normalize(test_reshape, axis=1)
validation_norm = tf.keras.utils.normalize(validation_reshape, axis=1)

    # train_norm = train_reshape
    # validation_norm = validation_reshape
    # test_norm = test_reshape


####################################
FC_SIZE = 1024
DTYPE = tf.float64


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


def network(input, labels_survival):

    xs = tf.placeholder(tf.float64, [None, 5, 64, 64, 3], name='input')
    channels = tf.placeholder(tf.float64, [None, 3], name='channels')
    labels_grad = tf.placeholder(tf.float64, [None, 2], name='grad_labels')
    labels_survival = tf.placeholder(tf.float64, [None, 3], name='survival_labels')
    loss = tf.placeholder(tf.float64)
    input = xs
    in_filters = channels

    with tf.variable_scope('conv1') as scope:
        out_filters = 16
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [1, out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

        input = conv1
        in_filters = out_filters

        pool1 = tf.nn.max_pool3d(input, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')  ############## 32*32*
        norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

        input = norm1

    with tf.variable_scope('conv2') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [1, out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

        input = conv2
        in_filters = out_filters

            # normalize input here
        input = tf.nn.max_pool3d(input, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #### 16*16*

    with tf.variable_scope('conv3_1') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [1, out_filters])
        bias = tf.nn.bias_add(conv, biases)
        input = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_2') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [1, out_filters])
        bias = tf.nn.bias_add(conv, biases)
        input = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_3') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [1, out_filters])
        bias = tf.nn.bias_add(conv, biases)
        input = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

        # normalize input here
        input = tf.nn.max_pool3d(input, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')  #### 8*8**


    with tf.variable_scope('local3') as scope:
        dim = np.prod(input.get_shape().as_list()[1:])
        input = tf.reshape(input, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local3 = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)

        input = local3

    with tf.variable_scope('local4') as scope:
        dim = np.prod(input.get_shape().as_list()[1:])
        input = tf.reshape(input, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local4 = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)

        input = local4

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(input.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, 3])
        biases = _bias_variable('biases', [3])
        softmax_linear = tf.add(tf.matmul(input, weights), biases, name=scope.name)
        # return softmax_linear

#############################################################################################################
    with tf.variable_scope('loss'):
        output_survival = tf.nn.softmax(softmax_linear)
        cross_entropy = tf.reduce_mean(labels_survival * tf.log(output_survival))
        loss = cross_entropy + 5e-4
        tf.summary.scalar('loss', loss)

    with tf.variable_scope('train'):
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
            5000,  # Decay step.
            0.98,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learning rate', self.learning_rate)

    self._train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=self.global_step)






tf.summary.scalar('my_loss', self.loss)
# def my_loss(validation_norm, y_validation_cat):
#     b=0.3
#     return tf.keras.backend.abs(losses.categorical_crossentropy(y_true=validation_norm, y_pred=y_validation_cat) - b) + b

my_optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = network(train_data, train_survival_labels_cat)
model.compile(loss=my_loss, optimizer=my_optimizer, metrics=['accuracy']) # We will measure the performance of the model using accuracy.
model_cnn = model.fit(train_norm, train_survival_labels_cat, epochs=50, validation_data=(validation_norm, validation_survival_labels_cat))
score = model.evaluate(test_norm, test_survival_labels_cat, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

#     # network.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])  # for two classes binary_crossentropy for more than 2 categorical_crossentropy
# tf.summary(network())
#     # network.summary()
# network.fit(train_data, train_survival_labels_cat, batch_size=32, epochs=100, verbose=1, shuffle=True, validation_data=(validation_data, validation_survival_labels_cat))  # A batch size(defult is 32) of 32 implies that we will compute the gradient and take a step in the direction of the gradient #   with a magnitude equal to the learning rate, after having pass 32 samples through the neural network
# test_loss, test_acc = network.evaluate(test_data, test_survival_labels_cat, verbose=0)
# print(test_loss, test_acc)
# print('Tested_Loss = ', test_loss)
# print('Tested_Accuracy = ', test_acc)
#
# print(network.history.keys())
# loss = network.history['loss']
# val_loss = network.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Loss_output_2D_3classes_Augm_vali(RotatedFolder)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('/exports/lkeb-hpc/gkarami/Code/Jobs/loss_2D_4.png')
# plt.show()



#########################################################################################333
