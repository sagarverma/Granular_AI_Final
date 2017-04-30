from os import listdir

root = '../../datasets/Rio/Rio_HGIS_Metro_extract/'
tars = listdir( root + 'imageChips/')

fout = open('extract_imageChips.sh','w')
i = 0
for tar in tars:
    fout.write('tar zxvf ' + root + 'imageChips/' + tar + ' -C ' + root + 'imageChips_extracted/ &\n')

    if i % 24 == 0:
        fout.write('wait\n')

    i += 1

fout.close()
