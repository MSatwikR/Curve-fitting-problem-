# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:10:30 2024

@author: Imprintec-Adithya
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\Messrechner\\Desktop\\Satwik\\Surface_defects_detection\\Inputfiles')
cur_dir=os.getcwd()
MP="MP001"

#Extracting information regarding the pixel of the height vorher and nachher file,scanStart and scanEnd 
f05=os.path.join(cur_dir,MP+"_height_vorher.txt")
with open(f05,"r") as hv:
    vorherLines=hv.readlines()
    for vLine in vorherLines:
        if("sxImg" in vLine): #Pixel information in x common for both vorher and nachher file
            pxv=vLine.replace(" ","")
            pxv=pxv.split("=")
            pix_x=int(pxv[-1])
        if("syImg" in vLine): #Pixel information in y common for both vorher and nachher file 
            pyv=vLine.replace(" ","")
            pyv=pyv.split("=")
            pix_y=int(pyv[-1])
        if("scanStart" in vLine):
            scanStart_v=vLine.replace(" ","")
            scanStart_v=scanStart_v.split("=")
            scanStart_v=float(scanStart_v[-1])
        if("scanEnd" in vLine):
            scanEnd_v=vLine.replace(" ","")
            scanEnd_v=scanEnd_v.split("=")
            scanEnd_v=float(scanEnd_v[-1])
            
        
        
            
f06=os.path.join(cur_dir,MP+"_height_nachher.txt")
with open(f06,"r") as hn:
    nachherLines=hn.readlines()
    for nLine in nachherLines:
        if("scanStart" in nLine):
            scanStart_n=nLine.replace(" ","")
            scanStart_n=scanStart_n.split("=")
            scanStart_n=float(scanStart_n[-1])
        if("scanEnd" in nLine):
            scanEnd_n=nLine.replace(" ","")
            scanEnd_n=scanEnd_n.split("=")
            scanEnd_n=float(scanEnd_n[-1])


#Data Extraction from imp files
#No Value Vorher
f01=os.path.join(cur_dir,MP+"_noValue_vorher.imp")
novalue_vorher=np.fromfile(f01,dtype=np.uint8) #reads Binary file 
novalue_vorher=np.reshape(novalue_vorher,(pix_x,pix_y)) #need pixel information for reshaping

#Height Vorher
f02=os.path.join(cur_dir,MP+"_height_vorher.imp")
height_vorher=np.fromfile(f02,dtype=np.uint16)
height_vorher=np.reshape(height_vorher,(pix_x,pix_y))
height_vorher=height_vorher/(2**16-1)*(scanEnd_v-scanStart_v)+scanStart_v

#No Value Nachher
f03=os.path.join(cur_dir,MP+"_noValue_nachher.imp")
novalue_nachher=np.fromfile(f03,dtype=np.uint8)
novalue_nachher=np.reshape(novalue_nachher, (pix_x,pix_y))

#Height Nachher
f04=os.path.join(cur_dir,MP+"_height_nachher.imp")
height_nachher=np.fromfile(f04,dtype=np.uint16)
height_nachher=np.reshape(height_nachher,(pix_x,pix_y))
height_nachher=height_nachher/(2**16-1)*(scanEnd_n-scanStart_n)+scanStart_n
                                                 
##Data Extraction completed

novalue_vorher=np.where(novalue_vorher<10,1,np.nan)
novalue_nachher=np.where(novalue_nachher<10,1,np.nan)

height_vorher=height_vorher*novalue_vorher
height_nachher=height_nachher*novalue_nachher

#Rescaling on the minimum Vorher value ignoring NaN's
rescale=np.nanmin(np.nanmin(height_vorher))
data_vorher=height_vorher-rescale
data_nachher=height_nachher-rescale

#Nachher - Vorher contains NaNs, this is the raw data
data_raw=data_nachher-data_vorher

print('Vorher data is',data_vorher,'\n')
print('Nachher data is',data_nachher,'\n')
print('raw data is',data_raw,'\n')

plt.imshow(height_nachher,cmap='Blues')
plt.colorbar()
plt.show()
plt.imshow(height_vorher,cmap='Greens')
plt.colorbar()
plt.show()







