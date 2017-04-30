from os import listdir
import numpy as np
import tifffile as tiff
import cv2

def stretch_8bit(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0 
        b = 255 
        c = np.percentile(bands[:,:,i], lower_percent)
        d = np.percentile(bands[:,:,i], higher_percent)        
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t
    return out.astype(np.uint8) 

out_img = np.zeros((2048+2048+908,2048+2048+779,3))

r1c1 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R1C1-055675869050_01_P001.TIF')
r1c1 = cv2.merge([r1c1[0],r1c1[1],r1c1[2]])
r1c1 = cv2.cvtColor(stretch_8bit(r1c1), cv2.COLOR_BGR2RGB)
print 'r1c1', r1c1.shape
cv2.imwrite('../../datasets/DC/r1c1.png', r1c1)

r1c2 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R1C2-055675869050_01_P001.TIF')
r1c2 = cv2.merge([r1c2[0],r1c2[1],r1c2[2]])
r1c2 = cv2.cvtColor(stretch_8bit(r1c2), cv2.COLOR_BGR2RGB)
print 'r1c2', r1c2.shape
cv2.imwrite('../../datasets/DC/r1c2.png', r1c2)

r1c3 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R1C3-055675869050_01_P001.TIF')
r1c3 = cv2.merge([r1c3[0],r1c3[1],r1c3[2]])
r1c3 = cv2.cvtColor(stretch_8bit(r1c3), cv2.COLOR_BGR2RGB)
print 'r1c3', r1c3.shape
cv2.imwrite('../../datasets/DC/r1c3.png', r1c3)

r2c1 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R2C1-055675869050_01_P001.TIF')
r2c1 = cv2.merge([r2c1[0],r2c1[1],r2c1[2]])
r2c1 = cv2.cvtColor(stretch_8bit(r2c1), cv2.COLOR_BGR2RGB)
print 'r2c1', r2c1.shape
cv2.imwrite('../../datasets/DC/r2c1.png', r2c1)

r2c2 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R2C2-055675869050_01_P001.TIF')
r2c2 = cv2.merge([r2c2[0],r2c2[1],r2c2[2]])
r2c2 = cv2.cvtColor(stretch_8bit(r2c2), cv2.COLOR_BGR2RGB)
print 'r2c2', r2c2.shape
cv2.imwrite('../../datasets/DC/r2c2.png', r2c2)

r2c3 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R2C3-055675869050_01_P001.TIF')
r2c3 = cv2.merge([r2c3[0],r2c3[1],r2c3[2]])
r2c3 = cv2.cvtColor(stretch_8bit(r2c3), cv2.COLOR_BGR2RGB)
print 'r2c3', r2c3.shape
cv2.imwrite('../../datasets/DC/r2c3.png', r2c3)

r3c1 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R3C1-055675869050_01_P001.TIF')
r3c1 = cv2.merge([r3c1[0],r3c1[1],r3c1[2]])
r3c1 = cv2.cvtColor(stretch_8bit(r3c1), cv2.COLOR_BGR2RGB)
print 'r3c1', r3c1.shape
cv2.imwrite('../../datasets/DC/r3c1.png', r3c1)

r3c2 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R3C2-055675869050_01_P001.TIF')
r3c2 = cv2.merge([r3c2[0],r3c2[1],r3c2[2]])
r3c2 = cv2.cvtColor(stretch_8bit(r3c2), cv2.COLOR_BGR2RGB)
print 'r3c2', r3c2.shape
cv2.imwrite('../../datasets/DC/r3c2.png', r3c2)

r3c3 = tiff.imread('../../datasets/DC/digitalglobe/WashingtonDC_Orthorectified/055675869050_01_P001_MUL/16AUG15155740-M3DS_R3C3-055675869050_01_P001.TIF')
r3c3 = cv2.merge([r3c3[0],r3c3[1],r3c3[2]])
r3c3 = cv2.cvtColor(stretch_8bit(r3c3), cv2.COLOR_BGR2RGB)
print 'r3c3', r3c3.shape
cv2.imwrite('../../datasets/DC/r3c3.png', r3c3)

out_img[0:2048,0:2048,:] = r1c1
out_img[0:2048,2048:2048*2,:] = r1c2
out_img[0:2048,2048*2:2048*2+777,:] = r1c3

out_img[2048:2048*2,0:2048,:] = r2c1
out_img[2048:2048*2,2048:2048*2,:] = r2c2
out_img[2048:2048*2,2048*2:2048*2+778,:] = r2c3


out_img[2048*2:2048*2+908,0:2048,:] = r3c1
out_img[2048*2:2048*2+908,2048:2048*2,:] = r3c2
out_img[2048*2:2048*2+907,2048*2:2048*2+779,:] = r3c3

cv2.imwrite('../../datasets/DC/DC.png', out_img)