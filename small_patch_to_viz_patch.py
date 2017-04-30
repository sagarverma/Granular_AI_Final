import pickle
import numpy as np
import cv2

dic = pickle.load(open('Rio.pkl','rb'))

for k in dic.keys():
	patch = dic[k]

	out_img = np.zeros((35*14,35*14,3))

	map_color = [(255,255,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,255)]

	for i in range(35):
		for j in range(35):
			if int(patch[i][j]) == -1:
				cv2.rectangle(out_img, (j*14,i*14), ((j+1)*14,(i+1)*32), (0,0,0), -1) 
			else:
				cv2.rectangle(out_img, (j*14,i*14), ((j+1)*14,(i+1)*32), map_color[int(patch[i][j])], -1)

	cv2.imwrite('../../outputs/Rio/ResNet/' + k, out_img)