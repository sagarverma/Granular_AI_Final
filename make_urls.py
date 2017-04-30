import csv
import matplotlib.pyplot as plt
import math
import urllib 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import StaleElementReferenceException
import threading
import time

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

def gen_lat_long(lat1, long1, distance, bearing):
    R = 6371e3 

    lat1 = math.radians(lat1) #Current lat point converted to radians
    long1 = math.radians(long1) #Current long point converted to radians
    bearing = math.radians(bearing)

    lat2 = math.asin( math.sin(lat1)*math.cos(distance/R) +
         math.cos(lat1)*math.sin(distance/R)*math.cos(bearing))

    long2 = long1 + math.atan2(math.sin(bearing)*math.sin(distance/R)*math.cos(lat1),
                 math.cos(distance/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    long2 = math.degrees(long2)

    return lat2, long2

#print gen_lat_long(28.62762,75.70129,135,90)
#print gen_lat_long(28.62762,75.70129,148,180)
#print gen_lat_long(28.62762,75.70129,1414.213562373095,135)

"""
fin = open('main_land.txt','r')
data = fin.read()

lat_long_eles = data.split(' ')

lat_long = []
lats = []
lons = []

for lle in lat_long_eles:
    lon_lat_ele = map(float, lle.split(','))

    lat = lon_lat_ele[1]
    lon = lon_lat_ele[0]

    print lat, lon
    lats.append(lat)
    lons.append(lon)

plt.scatter(lats,lons)
plt.show()
"""


count_tiles = 0
"""
def url_scraper(lat, lon, dist_east, dist_west, writer):

    global count_tiles

    fout = open(writer,'w')

    lat1, long1 = lat, lon #28.870697021484, 76.826019287109

    #print haversine(lat1, long1, lat2, long2)
    #print haversine(lat1, long1, lat3, long3)

    distance_east = dist_east #83340 
    distance_south = dist_west #49766 

    total_moves_east = int(math.ceil(distance_east / 135) * 5)
    total_moves_south = int(math.ceil(distance_south / 148) * 5)

    #print total_moves_south, total_moves_east

    driver = webdriver.PhantomJS()
    driver.get("http://bhuvan.nrsc.gov.in/map/bhuvan/bhuvan2d.php")

    driver.find_element_by_id("radio2").click()

    driver.find_element_by_id("Val").clear()
    driver.find_element_by_id("Val").send_keys(str(lat1) + ', ' + str(long1))
    driver.find_element_by_xpath('//img[@src = "img/searchnew.png"]').click()

    for i in range(8):
        driver.find_element_by_id(\
        "OpenLayers_Control_PanZoomBar_12_zoomin").click()

    #time.sleep(2)

    going_east = 1
    for i in range(total_moves_south):
        for j in range(total_moves_east):
            print count_tiles * 135 * 148 / 5000.0 / 5000.0
            #print 'The satellite has moved ' + str(j / 5.0 * 135) + 'm east and ' + str(i / 5.0 * 148) + 'm south'
            if going_east:
                try:
                    driver.find_element_by_id(\
                    "OpenLayers_Control_PanZoomBar_12_panright").click()
                    elems = driver.find_elements_by_class_name("olTileImage")
                
                    for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile1' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
                except StaleElementReferenceException as e:
                    time.sleep(2)
                    #driver.find_element_by_id(\
                    #"OpenLayers_Control_PanZoomBar_12_panleft").click()
                    pass

            else:
                try:
                    driver.find_element_by_id(\
                    "OpenLayers_Control_PanZoomBar_12_panleft").click()
                    elems = driver.find_elements_by_class_name("olTileImage")
                
                    for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile1' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
                except StaleElementReferenceException as e:
                    time.sleep(2)
                    #driver.find_element_by_id(\
                    #"OpenLayers_Control_PanZoomBar_12_panright").click()
                    pass

        for k in range(20):
            try:
                driver.find_element_by_id(\
                "OpenLayers_Control_PanZoomBar_12_pandown").click()
                elems = driver.find_elements_by_class_name("olTileImage")
            
                for elem in elems:
                    src = elem.get_attribute('src')
                    if 'tile1' in src:
                        fout.write(src + '\n')
                        count_tiles += 1
            except StaleElementReferenceException as e:
                time.sleep(2)
                #driver.find_element_by_id(\
                #"OpenLayers_Control_PanZoomBar_12_panup").click()
                pass
        
        if going_east:
            going_east = 0
        else:
            going_east = 1
"""

def fast_url_scraper(lat, lon, dist_east, dist_west, writer):

    global count_tiles

    fout = open(writer,'w')

    lat_iter, long_iter = lat, lon #28.870697021484, 76.826019287109

    #print haversine(lat1, long1, lat2, long2)
    #print haversine(lat1, long1, lat3, long3)

    distance_east = dist_east #83340 
    distance_south = dist_west #49766 

    #print total_moves_south, total_moves_east

    driver = webdriver.PhantomJS()
    driver.set_window_size(2000,2000)
    driver.get("http://bhuvan.nrsc.gov.in/map/bhuvan/bhuvan2d.php")

    driver.find_element_by_id("radio2").click()

    driver.find_element_by_id("Val").clear()
    driver.find_element_by_id("Val").send_keys(str(lat_iter) + ', ' + str(long_iter))
    driver.find_element_by_xpath('//img[@src = "img/searchnew.png"]').click()

    for i in range(8):
        driver.find_element_by_id(\
        "OpenLayers_Control_PanZoomBar_12_zoomin").click()

    #time.sleep(2)

    total_moves_east = int(math.ceil(distance_east / 1000.0 ))
    total_moves_south = int(math.ceil(distance_south / 1000.0 ))

    
    going_east = 1
    for i in range(total_moves_south):
        for j in range(total_moves_east):
            print writer, haversine(lat, lon, lat_iter, long_iter)
            if going_east:
                lat_iter, long_iter = gen_lat_long(lat_iter, long_iter, 1000, 90)
                driver.find_element_by_id("Val").clear()
                driver.find_element_by_id("Val").send_keys(str(lat_iter) + ', ' + str(long_iter))
                driver.find_element_by_xpath('//img[@src = "img/searchnew.png"]').click()
                try:
                    elems = driver.find_elements_by_class_name("olTileImage")

                    for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
                except StaleElementReferenceException as e:
                    time.sleep(5)
                    elems = driver.find_elements_by_class_name("olTileImage")

                    for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
                    pass
            else:
                lat_iter, long_iter = gen_lat_long(lat_iter, long_iter, 1000, 270)
                driver.find_element_by_id("Val").clear()
                driver.find_element_by_id("Val").send_keys(str(lat_iter) + ', ' + str(long_iter))
                driver.find_element_by_xpath('//img[@src = "img/searchnew.png"]').click()
                try:
                    elems = driver.find_elements_by_class_name("olTileImage")

                    for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
                except StaleElementReferenceException as e:
                    time.sleep(5)
                    for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
                    pass

        lat_iter, long_iter = gen_lat_long(lat_iter, long_iter, 1000, 180)
        driver.find_element_by_id("Val").clear()
        driver.find_element_by_id("Val").send_keys(str(lat_iter) + ', ' + str(long_iter))
        driver.find_element_by_xpath('//img[@src = "img/searchnew.png"]').click()
        try:
            elems = driver.find_elements_by_class_name("olTileImage")

            for elem in elems:
                src = elem.get_attribute('src')
                if 'tile' in src:
                    fout.write(src + '\n')
                    count_tiles += 1
        except StaleElementReferenceException as e:
            time.sleep(5)
            for elem in elems:
                        src = elem.get_attribute('src')
                        if 'tile' in src:
                            fout.write(src + '\n')
                            count_tiles += 1
            pass

        if going_east:
            going_east = 0
        else:
            going_east = 1
        
org_lat = 13.266830
org_long = 80.070987

org_dist_east = 32000
org_dist_south = 50000

THREADS = 44

for i in range(THREADS):
    this_lat, this_long = gen_lat_long(org_lat, org_long, org_dist_south / (1.0 * THREADS) * i, 180)
    print this_lat, this_long, org_lat, org_long, org_dist_south / (1.0 * THREADS) * i
    scrapper_thread = threading.Thread(target = fast_url_scraper, args = (this_lat, this_long, org_dist_east, org_dist_south / THREADS, 'Chennai/Chennai_urls' + str(i + 1) + '.txt'))
    scrapper_thread.start()
