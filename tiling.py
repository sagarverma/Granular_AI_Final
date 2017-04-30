from rtree import index
import csv
import numpy as np
from math import pi
import math
import pickle
p = index.Property()
idx = index.Index('Vizag.rtree', properties = p)

imgMap={}
i=1
j=2
c=0
lis =[]
latlon = []
ilist = []
#80.216674804688,12.837524414063,80.218048095703,12.838897705078
minLat= 1000.0
minLon= 1000.0
maxLat= 0.0
maxLon= 0.0
with open('Vizag_map.txt', 'rb') as csvfile:
	'''	
	for line in csvfile:
                line = line.strip().split(",")
                #print i,float(line[2]),float(line[1]),j
                idx.insert(i, (float(line[2]),float(line[1]),float(line[4]), float(line[3])),obj=j)
                imgMap[i]=line[0]
                #print imgMap[i]
                i=i+1
                j=j+1
		if minLat>float(line[2]):
			minLat = float(line[2])
		if maxLat<float(line[4]):
			maxLat = float(line[4])
		if minLon>float(line[1]):
                        minLon = float(line[1])
                if maxLon<float(line[3]):
                        maxLon = float(line[3])
	print minLat,minLon,maxLat,maxLon    	
	'''	    
	for line in csvfile:
		line = line.strip().split(",")
		imgMap[i]=line[0]
		i=i+1
	
	pickle.dump(imgMap,open("Vizag_imgMap.dat",'wb'))
	#21589.0, 1728.0, 16508.0
	#28.3392333984 76.8534851074 28.8954162598 77.606048584
	#print imgMap[21589], imgMap[1728], imgMap[16508]
	hits=idx.intersection((17.582244873,83.1184387207,17.931060791,83.5455322266), objects=True)
	for i in hits:
		ilist.append(i.id)
		#print i.id
		c = c+1
		lis = [t for t in i.bbox]
		if [i.id,lis[0],lis[1]] not in latlon:
			latlon.append([i.id,lis[0],lis[1]])
		
	#print c		
	latlon.sort(key=lambda x: (x[1:]))
 
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
def truncate(f, n):
    return np.floor(f * 10 ** n) / 10 ** n
ctr = 0
temp_arr = np.array(latlon)
#unique_items, counts = np.unique(temp_arr[:,1], return_counts=True)
#print len(unique_items), counts
r_earth=6378.137
imgList=[]
tempList = []
i=0
temp2_arr = np.zeros((350,1500))
lim=0
k=0
while (i < temp_arr.shape[0] and k < temp2_arr.shape[0]):
        lat00=temp_arr[i][1]
	#print lat00
        indices_lat00 = np.where(temp_arr[:,1]==lat00)[0]
	#print len(indices_lat00)
        for j in range (0,len(indices_lat00)):
		#print temp_arr[indices_lat00[j]]
		temp2_arr[k,j] = temp_arr[indices_lat00[j],0]
	
	#print temp2_arr[k,:]
	k = k + 1
	i = np.max(indices_lat00) + 1

#print temp_arr 
#print temp2_arr
#print imgMap[4031], imgMap[16911], imgMap[17853], imgMap[2872], imgMap[5118], imgMap[9903], imgMap[2659], imgMap[3748], imgMap[17128], imgMap[9583], imgMap[8407], imgMap[15695]

temp2_arr = temp2_arr[temp2_arr[:,0]!=0,:]
#print temp2_arr
#temp2_arr = temp2_arr[~np.all(temp2_arr == 0, axis=1)]
mergeTiles = []
for i in range(0,temp2_arr.shape[0]):
	for j in range(0,temp2_arr.shape[1],2):
		a = temp2_arr[i,j]
		if j+1<temp2_arr.shape[1]: 
			if temp2_arr[i,j+1]!=0:
				b = temp2_arr[i,j+1]
		if i+1<temp2_arr.shape[0]:
			if temp2_arr[i+1,j]!=0:
				c = temp2_arr[i+1,j]
		if i+1<temp2_arr.shape[0] and j+1<temp2_arr.shape[1]:
			if temp2_arr[i+1,j+1]!=0:
				d = temp2_arr[i+1,j+1] 	
		if not (a==0.0 and b==0 and c==0 and d==0):
			mergeTiles.append([a,b,c,d])
			print [a,b,c,d]
		a=0
		b=0
		c=0
		d=0

pickle.dump(mergeTiles,open("Vizag_tileInfo.dat",'wb'))
