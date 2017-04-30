from __future__ import division, print_function, absolute_import

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import image_preloader
from tflearn.optimizers import Momentum
import h5py
import numpy as np 
from scipy.misc import imread, imresize
import csv

tflearn.config.init_graph (num_cores=4, gpu_memory_fraction=0.8)

dataset_file = 'train.csv'
r = csv.reader(open(dataset_file,'r'), delimiter=' ')
X, y = image_preloader(dataset_file, image_shape=(320, 320),   mode='file', categorical_labels=True,   normalize=True)

img_aug = ImageAugmentation()
img_aug.add_random_crop((224,224))
img_aug.add_random_flip_leftright()
img_aug.add_random_90degrees_rotation()
img_aug.add_random_blur()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation()

inp = input_data(shape=[None, 224, 224, 3], data_augmentation=img_aug, name='input')

conv1_1 = conv_2d(inp, 64, 3, activation='relu', name="conv1_1", trainable=False)
conv1_2 = conv_2d(conv1_1, 64, 3, activation='relu', name="conv1_2", trainable=False)
pool1 = max_pool_2d(conv1_2, 2, strides=2)

conv2_1 = conv_2d(pool1, 128, 3, activation='relu', name="conv2_1", trainable=False)
conv2_2 = conv_2d(conv2_1, 128, 3, activation='relu', name= "conv2_2", trainable=False)
pool2 = max_pool_2d(conv2_2, 2, strides=2)

conv3_1 = conv_2d(pool2, 256, 3, activation='relu', name="conv3_1", trainable=False)
conv3_2 = conv_2d(conv3_1, 256, 3, activation='relu', name="conv3_2", trainable=False)
conv3_3 = conv_2d(conv3_2, 256, 3, activation='relu', name="conv3_3", trainable=False)
pool3 = max_pool_2d(conv3_3, 2, strides=2)

conv4_1 = conv_2d(pool3, 512, 3, activation='relu', name="conv4_1", trainable=False)
conv4_2 = conv_2d(conv4_1, 512, 3, activation='relu', name="conv4_2", trainable=False)
conv4_3 = conv_2d(conv4_2, 512, 3, activation='relu', name="conv4_3", trainable=False)
pool4 = max_pool_2d(conv4_3, 2, strides=2)

conv5_1 = conv_2d(pool4, 512, 3, activation='relu', name="conv5_1", trainable=False)
conv5_2 = conv_2d(conv5_1, 512, 3, activation='relu', name="conv5_2", trainable=False)
conv5_3 = conv_2d(conv5_2, 512, 3, activation='relu', name="conv5_3", trainable=False)
pool5 = max_pool_2d(conv5_3, 2, strides=2)

fc6 = fully_connected(pool5, 4096, activation='relu', name="fc6", trainable=False)
fc6_dropout = dropout(fc6, 0.5)

fc7 = fully_connected(fc6_dropout, 4096, activation='relu', name="fc7", trainable=False)
fc7_droptout = dropout(fc7, 0.5)

fc8 = fully_connected(fc7_droptout, 30, activation='softmax', name="fc8")

mm = Momentum(learning_rate=0.01, momentum=0.9, lr_decay=0.1, decay_step=1000)

network = regression(fc8, optimizer=mm , loss='categorical_crossentropy', restore=False)

print("Network defined.")
model = tflearn.DNN(network, checkpoint_path='../../checkpoints/vgg16_AID', max_checkpoints=1, tensorboard_verbose=3)

print("Model defined.")
"""
print(model.get_weights(conv1_1.W).shape)
print(model.get_weights(conv1_2.W).shape)
print(model.get_weights(conv2_1.W).shape)
print(model.get_weights(conv2_2.W).shape)
print(model.get_weights(conv3_1.W).shape)
print(model.get_weights(conv3_2.W).shape)
print(model.get_weights(conv3_3.W).shape)
print(model.get_weights(conv4_1.W).shape)
print(model.get_weights(conv4_2.W).shape)
print(model.get_weights(conv4_3.W).shape)
print(model.get_weights(conv5_1.W).shape)
print(model.get_weights(conv5_2.W).shape)
print(model.get_weights(conv5_3.W).shape)
print(model.get_weights(fc6.W).shape)
print(model.get_weights(fc7.W).shape)
print(model.get_weights(fc8.W).shape)
"""

"""
print("Start intiallizing weights")
weights = np.load("/media/Drive2/sagar/ImageNet/weights/vgg16_weights.npz", "r")

model.set_weights(conv1_1.W, weights["conv1_1_W"])
model.set_weights(conv1_1.b, weights["conv1_1_b"])

model.set_weights(conv1_2.W, weights["conv1_2_W"])
model.set_weights(conv1_2.b, weights["conv1_2_b"])

model.set_weights(conv2_1.W, weights["conv2_1_W"])
model.set_weights(conv2_1.b, weights["conv2_1_b"])

model.set_weights(conv2_2.W, weights["conv2_2_W"])
model.set_weights(conv2_2.b, weights["conv2_2_b"])

model.set_weights(conv3_1.W, weights["conv3_1_W"])
model.set_weights(conv3_1.b, weights["conv3_1_b"])

model.set_weights(conv3_2.W, weights["conv3_2_W"])
model.set_weights(conv3_2.b, weights["conv3_2_b"])

model.set_weights(conv3_3.W, weights["conv3_3_W"])
model.set_weights(conv3_3.b, weights["conv3_3_b"])

model.set_weights(conv4_1.W, weights["conv4_1_W"])
model.set_weights(conv4_1.b, weights["conv4_1_b"])

model.set_weights(conv4_2.W, weights["conv4_2_W"])
model.set_weights(conv4_2.b, weights["conv4_2_b"])

model.set_weights(conv4_3.W, weights["conv4_3_W"])
model.set_weights(conv4_3.b, weights["conv4_3_b"])

model.set_weights(conv5_1.W, weights["conv5_1_W"])
model.set_weights(conv5_1.b, weights["conv5_1_b"])

model.set_weights(conv5_2.W, weights["conv5_2_W"])
model.set_weights(conv5_2.b, weights["conv5_2_b"])

model.set_weights(conv5_3.W, weights["conv5_3_W"])
model.set_weights(conv5_3.b, weights["conv5_3_b"])
model.set_weights(fc6.W, weights["fc6_W"])
model.set_weights(fc6.b, weights["fc6_b"])

model.set_weights(fc7.W, weights["fc7_W"])
model.set_weights(fc7.b, weights["fc7_b"])

#model.set_weights(fc8.W, weights["fc8_W"])
#model.set_weights(fc8.b, weights["fc8_b"])


print("Setting weights done.")
"""

model.load('../../checkpoints/vgg16_AID-942')

model.fit(X, y, n_epoch=74, shuffle=True, show_metric=True, batch_size=128, snapshot_step=500, run_id='vgg16_AID')


