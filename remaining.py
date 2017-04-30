import csv
from bs4 import BeautifulSoup
from os import listdir

all_awifs = listdir('../../datasets/AWiFS')

dones = {}

for aw in all_awifs:
    dones[aw[-17:-4].replace('_','')] = 1

fin = open('AWiFS.html','r')
data = fin.read()

files = []
soup = BeautifulSoup(data, 'lxml')
table =  soup.find_all('table')[35]
for tr in table.find_all('tr')[2:]:
    tds = tr.find_all('td')
    temp = [elem.text.encode('utf-8') for elem in tds]
    files.append([temp[1].replace(' ','').lower(), temp[2].replace(' ',''), temp[3].replace(' ','').replace('Ver:','v').replace('R','r').replace('v1','')])

url = 'http://bhuvan.nrsc.gov.in/data/download/tools/download1/downloadlink.php?id=aw' 
end_url = '&se=AWIFS'


not_done = []
for file in files:
    if file[0] + file[2] not in dones:
        not_done.append(url + file[0] + file[2] + end_url)
        print url + file[0] + file[2] + end_url
