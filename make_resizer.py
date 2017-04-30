from os import listdir

fout = open('resize_batch.sh','w')

root = '../../datasets/AID/AID_imgs/'

folders = listdir(root)


for f in folders:
    fout.write('for name in ' + root + '/' + f + '/' + '*.jpg; do     convert -resize 320x320\! $name $name; done\n')

fout.close()
