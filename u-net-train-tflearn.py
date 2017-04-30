import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.optimizers import Momentum
import numpy as np
import cv2
from os import listdir
import tensorflow as tf 

def jaccard_coef(y_true, y_pred):
    intersection = tf.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = tf.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return tf.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = tf.round(K.clip(y_pred, 0, 1))

    intersection = tf.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return tf.mean(jac)

def to_categorical(y, nb_classes):
    w, h = y.shape[0], y.shape[1]
    Y = np.zeros((w * h, nb_classes))
    y = np.reshape(y, (w * h))
    Y[np.arange(w*h),y] = 1.
    Y = np.reshape(Y, (w, h, nb_classes))
    return Y

class Preloader(object):
    def __init__(self, array, function):
        self.array = array
        self.function = function

    def __getitem__(self, id):
        if type(id) in [list, np.ndarray]:
            return [self.function(self.array[i]) for i in id]
        elif isinstance(id, slice):
            return [self.function(arr) for arr in self.array[id]]
        else:
            return self.function(self.array[id])

    def __len__(self):
        return len(self.array)

class ImagePreloader(Preloader):
    def __init__(self, array):
        fn = lambda x: self.preload(x)
        super(ImagePreloader, self).__init__(array, fn)

    def preload(self, path):
        image = cv2.imread(path)
        return image

class LabelPreloader(Preloader):
    def __init__(self, array, n_class=None):
        fn = lambda x: self.preload(x, n_class)
        super(LabelPreloader, self).__init__(array, fn)

    def preload(self, path, n_class):
        label = cv2.imread(path, 0)
        return to_categorical(label, n_class)

def image_preloader(target_path, n_classes=12):
    files = listdir(target_path)

    images, labels = [], []
    for filename in files:
        if 'mask' in filename:
            images.append(target_path + filename[:-9] + '.png')
            labels.append(target_path + filename)

    X = ImagePreloader(images)
    Y = LabelPreloader(labels, n_classes)

    return X, Y

n_classes = 12

tflearn.config.init_graph (num_cores=12, gpu_memory_fraction=0.2)

train_dataset_folder = '../../datasets/DSTL/train_sample_three_band/'
X, y = image_preloader(train_dataset_folder, n_classes)

inp = input_data(shape=[None, 8, 160, 160], name="input")
conv1_1 = conv_2d(inp, 32, 3, 3, activation='relu', name="conv1_1")
conv1_2 = conv_2d(conv1_1, 32, 3, 3, activation='relu', name="conv1_2")
pool1 = max_pool_2d(conv1_2, 2)

conv2_1 = conv_2d(pool1, 64, 3, 3, activation='relu', name="conv2_1")
conv2_2 = conv_2d(conv2_1, 64, 3, 3, activation='relu', name="conv2_2")
pool2 = max_pool_2d(conv2_2, 2)

conv3_1 = conv_2d(pool2, 128, 3, 3, activation='relu', name="conv3_1")
conv3_2 = conv_2d(conv3_1, 128, 3, 3, activation='relu', name="conv3_2")
pool3 = max_pool_2d(conv3_2, 2)

conv4_1 = conv_2d(pool3, 256, 3, 3, activation='relu', name="conv4_1")
conv4_2 = conv_2d(conv4_1, 256, 3, 3, activation='relu', name="conv4_2")
pool4 = max_pool_2d(conv4_2, 2)

conv5_1 = conv_2d(pool4, 512, 3, 3, activation='relu', name="conv5_1")
conv5_2 = conv_2d(conv5_1, 512, 3, 3, activation='relu', name="conv5_2")

up6 = merge([upsample_2d(conv5_2, 2), conv4_2], mode='concat', axis=1, name='upsamle-5-merge-4')
conv6_1 = conv_2d(up6, 256, 3, 3, activation='relu', name="conv6_1")
conv6_2 = conv_2d(conv6_1, 256, 3, 3, activation='relu', name="conv6_2")

up7 = merge([upsample_2d(conv6_2, 2), conv3_2], mode='concat', axis=1, name='upsamle-6-merge-3')
conv7_1 = conv_2d(up7, 128, 3, 3, activation='relu', name="conv7_1")
conv7_2 = conv_2d(conv7_1, 128, 3, 3, activation='relu', name="conv7_2")

up8 = merge([upsample_2d(conv7_2, 2), conv2_2], mode='concat', axis=1, name='upsamle-7-merge-2')        
conv8_1 = conv_2d(up8, 64, 3, 3, activation='relu', name="conv8_1")
conv8_2 = conv_2d(conv8_1, 64, 3, 3, activation='relu', name="conv8_2")

up9 = merge([upsample_2d(conv8_2, 2), conv1_2], mode='concat', axis=1, name='upsamle-8-merge-1')
conv9_1 = conv_2d(up9, 32, 3, 3, activation='relu', name="conv9_1")
conv9_2 = conv_2d(conv9_1, 32, 3, 3, activation='relu', name="conv9_2")

conv10 = conv_2d(conv9_2, n_classes, 1, 1, activation='sigmoid', name="conv10")

network = regression(conv10, optimizer='Adam' , loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

model.fit(X, y, validation_set=0.2, n_epoch=10, shuffle=True, show_metric=True, batch_size=128, snapshot_step=100, run_id='U-Net-DSTL')
