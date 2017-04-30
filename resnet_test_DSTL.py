
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tflearn
import numpy as np
import cv2
import Image
import csv
from os import listdir
from tifffile import imread
from math import ceil, log
import pickle 

class_map = {0:'building', 1:'barren_land',2:'trees',3:'grassland',4:'road',5:'water'}

tflearn.config.init_graph (num_cores=1, gpu_memory_fraction=0.3)

# Residual blocks
# 256 layers: n=5, 256 layers: n=9, 110 layers: n=18
n = 5

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center([112.3256420256,1256.6802564592,1256.1983022562], per_channel=True)


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
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=256000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='../../checkpoints/resnet_sat6',
                    max_checkpoints=1, tensorboard_verbose=3)

"""
model.fit(X, y, n_epoch=200, validation_set=(X_test, y_test),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=2256, shuffle=True,
          run_id='resnet_sat6')
"""

model.load('../../checkpoints/resnet_sat/resnet_sat6-3500')

"""
r = csv.reader(open('../utilities/Chandigarh_map.txt', 'r'))
w = csv.writer(open('../../outputs/Chandigarh_patch_classes.csv','w'))

in_path = listdir('../../datasets/Chandigarh_imgs/')

those_that_have_lived = {}

for img_name in in_path:
    those_that_have_lived[img_name] = 1

for row in r:
    if row[0] in those_that_have_lived:
        img = Image.open('../../datasets/Chandigarh_imgs/' + row[0])
        img = img.resize((256,256), Image.ANTIALIAS)
        img = np.asarray(img, dtype=np.float256)
        img /= 255.

        class_this = np.argmax(model.predict([img]))

        w.writerow([class_map[class_this]] + row[1:])
"""

"""
images = listdir('../../datasets/Chandigarh_imgs')
images.sort()


w = csv.writer(open('Chandigarh_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Chandigarh_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Chennai_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Chennai_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Chennai_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Delhi_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Delhi_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Delhi_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Mumbai_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Mumbai_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Mumbai_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Vizag_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Vizag_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Vizag_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Puducherry_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Puducherry_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Puducherry_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Hyderabad_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Hyderabad_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Hyderabad_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

images = listdir('../../datasets/Bengaluru_imgs')
images.sort()

map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

w = csv.writer(open('Bengaluru_patch_classes.csv','wb'))

for image in images:
    img = cv2.imread('../../datasets/Bengaluru_imgs/' + image)
    if not img is None:
        img = img.astype(np.float32)
        class_this = np.argmax(model.predict([img]))
        w.writerow([image, class_this])

"""
map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

images = listdir('../../datasets/Vizag_imgs')
images.sort()

out_map = {}

for image in images:
	img = cv2.imread('../../datasets/Vizag_imgs/' + image)
	if not img is None:
		img = img.astype(np.float32)

		all_solutions = []

		out_img = np.zeros((256,256,3))

		tot = 0
		classes_tot = [0,0,0,0,0,0]

		ins = []
		in_map_ij = {}

		in_no = 0
		for i in range(0,256,8):
			for j in range(0,256,8):
				if i + 32 < 256 and j + 32 < 256:
					ins.append(cv2.resize(img[i:i+32,j:j+32], (28,28)))
					in_map_ij[in_no] = [i,j]
					in_no += 1

		preds = model.predict(ins)

		pred_mat = np.ones((32,32))
		pred_mat *= -1

		for in_no in range(len(preds)):
			pred = preds[in_no]
			for k in range(6):
				if preds[in_no][k] > 0.90:
					#cv2.rectangle(out_img, (j,i), (j+32,i+32), map_color[k], -1) 
					pred_mat[in_map_ij[in_no][0]/8][in_map_ij[in_no][1]/8] = k			
					break

		out_map[image] = pred_mat

		print image

fout = open('Vizag_patches.pkl','wb')
pickle.dump(out_map, fout)
