from rtree import index
import csv
import numpy as np
from math import pi
import math
import pickle
p = index.Property()
idx = index.Index('chennai.rtree', properties = p)
#print idx

#idx.insert(4321, (34.3776829412, 26.7375853734, 49.3776829412, 41.7375853734))
#idx.insert(2207, (30.6683349609, 76.7353820801, 34.5657287812, 76.982773712))
imgMap={}
i=1
j=2
c=0
lis =[]
latlon = []
ilist = []
with open('Chennai_map.txt', 'rb') as csvfile:
        
	for line in csvfile:
		line = line.strip().split(",")
		#print i,float(line[2]),float(line[1]),j
		idx.insert(i, (float(line[2]),float(line[1]),float(line[4]), float(line[3])),obj=j)
		imgMap[i]=line[0]
		print i
		#print imgMap[i]
		i=i+1
		j=j+1
        
	hits=idx.intersection((30.5735778809,76.7106628418,30.8001708984 ,76.9537353516), objects=True)
	for i in hits:
		ilist.append(i.id)
		#print i.id
		c = c+1
		lis = [t for t in i.bbox]
		latlon.append([i.id,lis[0],lis[1]])
		
	#print c		
	latlon.sort(key=lambda x: (x[1:]))
	#for l in latlon:
		#print imgMap[l[0]]
	print np.min(lat), np.max(lat), np.min(lon), np.max(lon)


 
#print latlon
def measure(lat1, lon1, lat2, lon2):
    r = 6378.137
    dLat = (lat2 - lat1) * math.pi / 180
    dLon = (lon2 - lon1) * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) +math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) *math.sin(dLon/2)*math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = r * c
    return d

#width = measure(latlon[1][1],latlon[1][2],latlon[0][1],latlon[0][2])
width = 0.131620997939
height = 0.152874056612
lat0= latlon[0][1]
lon0= latlon[0][2]
#print len(latlon)

'''
height = 0
for l in latlon:
	if l[2] == lon00 and l[1]!=lat00:
		#print l[2], l[1]
		height =  measure(l[1],l[2],lat00,lon00)
 		break
print width, height

temp = []
ctr = 0
ix = 0
last = 22076
for i in range(0,len(latlon)-1):
	
	lat00=latlon[i][1]
	ix = i
	if ix >= len(latlon)-1:
		#print "inside outer if"
		break
	while (ix < len(latlon)-1 and latlon[ix+1][1] == lat00):
		ctr = ctr+1
		if (measure(latlon[ix+1][1],latlon[ix+1][2],latlon[ix][1],latlon[ix][2])) > width:
			temp.append([lat00,latlon[ix],latlon[ix+1]])
		ix = ix + 1
	i = ctr	
 				
#print len(temp)
'''

def truncate(f, n):
    return np.floor(f * 10 ** n) / 10 ** n
ctr = 0
temp_arr = np.array(latlon)
unique_items, counts = np.unique(temp_arr[:,1], return_counts=True)
print len(unique_items), counts
#print temp_arr
#id_map = {}
r_earth=6378.137
#indices = [np.where(temp_arr[:,1]==lat0)[0]]
#print type(indices)
imgList=[]
tempList = []
k=0
i=0
temp2_arr = np.zeros((350,1500))
while (i < temp_arr.shape[0]):
	lat00=temp_arr[i][1]
	#print lat00
	lat10=lat00  + (height / r_earth) * (180 / math.pi)
	lat10 = truncate(lat10,4)
	#lon10=lon00 + (width / r_earth) * (180 / math.pi) / math.cos(lat00 * pi/180)
	indices_lat00 = np.where(temp_arr[:,1]==lat00)[0]
	indices_lat10 = np.where(truncate(temp_arr[:,1],4)==lat10)[0]
	#print len(indices_lat10)
	#print len(indices_lat00)
	if indices_lat10.shape[0]!=0:
		k = indices_lat10[-1]
		i = i+k+1
		print "k"
		#ctr = min(indices_lat10.shape[0],indices_lat00.shape[0])
		#print temp_arr[indices_lat00[7]]
		while(j<=ctr-2):
                	imgList.append([temp_arr[indices_lat00[j]],temp_arr[indices_lat00[j+1]],temp_arr[indices_lat10[j]],temp_arr[indices_lat10[j+1]]])
			j= j+2
	else:
		if lat10 not in tempList:
			tempList.append(lat10)
		i=i+1
	#print i
print tempList, len(tempList)
#print imgList

'''
missing = []
lastLon = 76.788940429688
for item in tempList:
	missing.append([item,lon0])
	lon0 = lon0 + (width / r_earth) * (180 / math.pi) / math.cos(lat00 * pi/180)

print missing, len(missing)
#print len(imgList), len(latlon)/4
'''
#print imgMap[17666], imgMap[12421], imgMap[336], imgMap[16348]
