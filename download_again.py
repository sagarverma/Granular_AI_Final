import csv
import re
import math 
import urllib

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

r1 = csv.reader(open('Mumbai_map.txt','r'))
fin1 = open('Mumbai_valid','r')

fin2 = open('Mumbai.txt','r')

dic = {}
i = 0
for line in fin2.readlines():
    lats_longs = re.search('BBOX=(.*?)&WIDTH', line, re.IGNORECASE).group(1)
    lats_longs_parsed = map(float, lats_longs.split(','))
    distance, _ = haversine(lats_longs_parsed[1], lats_longs_parsed[0], lats_longs_parsed[3], lats_longs_parsed[2])
    if lats_longs not in dic and distance < 350:
        dic[lats_longs] = [i, line]
        i += 1

valids = {}
for line in fin1.readlines():
    img_name = re.search('(.*?).jpg', line, re.IGNORECASE).group(1)
    valids[img_name + '.jpg'] = 1


invalids = []
for row in r1:
    if row[0] not in valids:
        invalids.append([row[0], dic[row[1]+','+row[2]+','+row[3]+','+row[4]][1]])

print len(invalids)
for invalid in invalids:
    url = invalid[1]
    i = invalid[0]
    urllib.urlretrieve (url, "../../datasets/Mumbai_imgs/" + i)
