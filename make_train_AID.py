from os import listdir
import csv

w1 = csv.writer(open('../VGG16/train.csv','w'), delimiter=' ')
w2 = csv.writer(open('../VGG16/class_map.csv','w'), delimiter=' ')

classes = listdir('../../datasets/AID/AID_imgs/')

i = 0

for clas in classes:
    w2.writerow([clas, i])

    imgs = listdir('../../datasets/AID/AID_imgs/' + clas + '/')

    for img in imgs:
        w1.writerow(['../../datasets/AID/AID_imgs/' + clas + '/' + img, i])

    i += 1


