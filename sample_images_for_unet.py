from os import listdir
import numpy as np 
import cv2
from math import ceil 

images = listdir('../../datasets/DSTL/train_three_band/')

sample_no = 0

for image in images:
    if 'mask' in image:
        img = cv2.imread('../../datasets/DSTL/train_three_band/' + image[:-9] + '.png')
        limg = cv2.imread('../../datasets/DSTL/train_three_band/' + image, 0)

        img = cv2.resize(img, (int(ceil(img.shape[0]/232)*232), int(ceil(img.shape[1]/232)*232)))
        limg = cv2.resize(limg, (int(ceil(limg.shape[0]/232)*232), int(ceil(limg.shape[1]/232)*232)))

        for i in range(0,img.shape[0],232):
            for j in range(0,img.shape[1],232):
                if np.count_nonzero(limg[i:i+232,j:j+232]) > 0:
                    cv2.imwrite('../../datasets/DSTL/train_sample_three_band/' + str(sample_no).zfill(10) + '.png', img[i:i+232,j:j+232])
                    cv2.imwrite('../../datasets/DSTL/train_sample_three_band/' + str(sample_no).zfill(10) + '_mask.png', limg[i:i+232,j:j+232])
                    sample_no += 1
