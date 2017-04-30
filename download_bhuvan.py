import urllib
import csv
import re
import threading
from itertools import islice
import math
from os import listdir, path
#import resource

#resource.setrlimit(resource.RLIMIT_NOFILE, (2048,4096))

def haversine(lat1,lon1,lat2,lon2):
    R = 6371e3
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delphi = math.radians(lat2-lat1)
    dellambda = math.radians(lon2-lon1)

    a = math.sin(delphi/2) * math.sin(delphi/2) + math.cos(phi1) * math.cos(phi2) * math.sin(dellambda/2) * math.sin(dellambda/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    d = R * c

    y = math.sin(dellambda) * math.cos(phi2);
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dellambda);
    brng = math.degrees(math.atan2(y, x))
    brng = (brng+360) % 360
    return d, brng

fin = open('Vizag.txt','r')
w = csv.writer(open('Vizag_map1.txt','w'))

dic = {}
i = 0
for line in fin.readlines():
    lats_longs = re.search('BBOX=(.*?)&WIDTH', line, re.IGNORECASE).group(1)
    lats_longs_parsed = map(float, lats_longs.split(','))
    distance, _ = haversine(lats_longs_parsed[1], lats_longs_parsed[0], lats_longs_parsed[3], lats_longs_parsed[2])
    if lats_longs not in dic and distance < 350:
        dic[lats_longs] = [i, line]
        i += 1

print len(dic.keys())

def download_images(dic):
    global w
    for lats_longs in dic.keys():
        url = dic[lats_longs][1]
        i = dic[lats_longs][0]
        if not path.exists("../../datasets/Vizag_imgs/" + str(i).zfill(10) + ".jpg"):
            lats_longs_parsed = map(float, lats_longs.split(','))
            w.writerow([str(i).zfill(10) + ".jpg"] + lats_longs_parsed)
            urllib.urlretrieve (url, "../../datasets/Vizag_imgs/" + str(i).zfill(10) + ".jpg")

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in xrange(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

THREADS = 300


i = 0
for dic_slice in list(chunks(dic, len(dic.keys()) / THREADS + 1)):
    #print dic_slice
    scrapper_thread = threading.Thread(target = download_images, args = (dic_slice,))
    scrapper_thread.start()
