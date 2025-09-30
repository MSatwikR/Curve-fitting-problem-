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
MP="MP002"

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
        if ("SensoraufloesungX" in vLine):
            sensor_res_x = vLine.replace(" ", "")
            sensor_res_x = sensor_res_x.split("=")
            #sensor_res_x = sensor_res_x[-1].replace(".", ",")
            sensor_res_x = float(sensor_res_x[-1])
        if ("SensoraufloesungY" in vLine):
            sensor_res_y = vLine.replace(" ", "")
            sensor_res_y = sensor_res_y.split("=")
            sensor_res_y = float(sensor_res_y[-1])
            
        
        
            
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

novalue_vorher=np.where(novalue_vorher<129,1,np.nan)
novalue_nachher=np.where(novalue_nachher<129,1,np.nan)

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

# Calculate actual physical dimensions from sensor resolution
# From your files: SensoraufloesungX = 2.8334 μm/pixel, SensoraufloesungY = 2.8337 μm/pixel
# Image size: 1920 x 1920 pixels
scan_width_mm = (pix_x * sensor_res_x) / 1000  # Convert μm to mm
scan_height_mm = (pix_y * sensor_res_y) / 1000  # Convert μm to mm

print(f"Actual scan dimensions: {scan_width_mm:.2f} mm x {scan_height_mm:.2f} mm")
print(f"Pixel resolution: {sensor_res_x:.3f} μm/pixel (X), {sensor_res_y:.3f} μm/pixel (Y)")

# FIRST SEPARATE PLOT: Height Nachher with plasma colormap
plt.figure(figsize=(8, 6))
im1 = plt.imshow(height_nachher, cmap='plasma',
                 extent=[0, scan_width_mm,0, scan_height_mm],
                 aspect='equal', origin='upper')

# Add colorbar for first plot
cbar1 = plt.colorbar(im1, shrink=0.8)
cbar1.set_label('[μm]', rotation=0, labelpad=20, fontsize=12)
cbar1.formatter.set_powerlimits((0, 0))  # Force scientific notation if needed
cbar1.update_ticks()

# Format first plot
plt.xlabel('[mm]', fontsize=12)
plt.ylabel('[mm]', fontsize=12)
plt.title('Height Nachher', fontsize=14)
plt.tight_layout()
plt.show()

# SECOND SEPARATE PLOT: Height Vorher with viridis colormap
plt.figure(figsize=(8, 6))
im2 = plt.imshow(height_vorher, cmap='viridis',
                 extent=[0, scan_width_mm,0, scan_height_mm],
                 aspect='equal', origin='upper')

# Add colorbar for second plot
cbar2 = plt.colorbar(im2, shrink=0.8)
cbar2.set_label('[μm]', rotation=0, labelpad=20, fontsize=12)
cbar2.formatter.set_powerlimits((0, 0))  # Force scientific notation if needed
cbar2.update_ticks()

# Format second plot
plt.xlabel('[mm]', fontsize=12)
plt.ylabel('[mm]', fontsize=12)
plt.title('Height Vorher', fontsize=14)
plt.tight_layout()
plt.show()


#for the croped image:

'''plt.imshow(height_nachher,cmap='plasma')
plt.colorbar()
plt.show()
plt.imshow(height_vorher,cmap='viridis')
plt.colorbar()
plt.show()'''


def center_crop_and_plot(data, crop_fraction, title):
    height, width = data.shape

    # Calculate crop indices
    center_y, center_x = height // 2, width // 2
    crop_size_y = int(height * crop_fraction)
    crop_size_x = int(width * crop_fraction)

    half_crop_y = crop_size_y // 2
    half_crop_x = crop_size_x // 2

    # Crop boundaries
    start_y = center_y - half_crop_y
    end_y = center_y + half_crop_y
    start_x = center_x - half_crop_x
    end_x = center_x + half_crop_x

    # Crop the data
    cropped_data = data[start_y:end_y, start_x:end_x]

    # Calculate physical dimensions
    crop_width_mm = (crop_size_x * sensor_res_x) / 1000
    crop_height_mm = (crop_size_y * sensor_res_y) / 1000

    print(f"Original size: {height} x {width} pixels")
    print(f"Cropped size: {crop_size_y} x {crop_size_x} pixels")
    print(f"Physical dimensions: {crop_width_mm:.2f} x {crop_height_mm:.2f} mm")

    # Plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cropped_data, cmap='plasma',
                    extent=[0, crop_width_mm, crop_height_mm, 0],
                    aspect='equal', origin='upper')

    plt.colorbar(im, label='[μm]')
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.title(f'{title} - Center {int(crop_fraction * 100)}%')
    plt.tight_layout()
    plt.show()

    return cropped_data


height_nachher_center = center_crop_and_plot(height_nachher,crop_fraction=0.1,title="Height Nachher")

height_vorher_center = center_crop_and_plot(height_vorher,crop_fraction=0.1,title="Height Nachher")






