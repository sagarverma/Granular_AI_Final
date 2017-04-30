import csv
import cv2
import math
import numpy as np 
import matplotlib.pyplot as plt  

r = csv.reader(open('Chandigarh_map.txt', 'r'))


all_lat_lon = []

for row in r:
    all_lat_lon.append([float(row[2]),float(row[1]), row[0]])


all_lat_lon.sort(key = lambda x: (-x[0], x[1]))

zeros = np.zeros((256*200, 256*200))

start_lat = all_lat_lon[0][0]

curr_x = 0
curr_y = 0

i = 0
for lat_lon in all_lat_lon:
    if start_lat != lat_lon[0]:
        start_lat = lat_lon[0]
        curr_y += 256
        curr_x = 0
        img = cv2.imread('Chandigarh_imgs/' + lat_lon[2], 0)
        zeros[curr_y:curr_y+256,curr_x:curr_x+256] = img

        curr_x += 256
    else:
        img = cv2.imread('Chandigarh_imgs/' + lat_lon[2], 0)
        zeros[curr_y:curr_y+256,curr_x:curr_x+256] = img

        curr_x += 256

zeros_small = cv2.resize(zeros, (1000,1000))
cv2.imwrite('Chandigarh_small.png',zeros_small)