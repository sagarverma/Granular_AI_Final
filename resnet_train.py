from __future__ import division, print_function, absolute_import

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tflearn
import numpy as np
from scipy.io import loadmat

tflearn.config.init_graph (num_cores=4, gpu_memory_fraction=0.5)

train_dat = loadmat('../../datasets/SAT/sat-6-full.mat')
X = np.rollaxis(train_dat['train_x'], 3).astype(np.float32)[:,:,:,0:3]
y = np.rollaxis(train_dat['train_y'], 1)

X_test = np.rollaxis(train_dat['test_x'], 3).astype(np.float32)[:,:,:,0:3]
y_test = np.rollaxis(train_dat['test_y'], 1)

print('Data loaded')
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center([112.3404207,114.68074592,114.19830272], per_channel=True)


# Building Residual Network
net = tflearn.input_data(shape=[None, 28, 28, 3], data_preprocessing=img_prep)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, 6, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='../../checkpoints/resnet_sat/resnet_sat6',
                    max_checkpoints=1, tensorboard_verbose=3)

#model.load('../../checkpoints/resnet_sat6-4500')
model.fit(X, y, n_epoch=1, validation_set=(X_test, y_test),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=256, shuffle=True,
          run_id='resnet_sat6')

#print(model.evaluate(X_test, y_test, batch_size=512))
