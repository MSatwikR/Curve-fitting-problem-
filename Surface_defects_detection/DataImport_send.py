# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:10:30 2024

@author: Imprintec-Adithya
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import for_one_file

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

scan_data = [scan_width_mm, scan_height_mm]
scan_data = np.array(scan_data)
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



def circle_fit(x, y, radius,tol1, tol2):
    if (x**2 + y**2) in range(radius**2 + tol1, radius**2 + tol1):
        print("Nachher height is just satisfactory")
    elif (x**2 + y**2) in range(radius**2 + tol2, radius**2 - tol2):
        print("Nachher height is good")
    else:
        print("Nachher height is schlecht")




def detect_indentation_opencv(height_data, min_radius_mm, max_radius_mm,sensor_resolution_um_per_px):
    # Convert to 8-bit image for OpenCV
    # Normalize and handle NaN values
    data_clean = np.nan_to_num(height_data, nan=np.nanmean(height_data))
    data_norm = ((data_clean - np.min(data_clean)) /
                 (np.max(data_clean) - np.min(data_clean)) * 255).astype(np.uint8)

    # Convert pixel radii range
    pixel_size_mm = sensor_resolution_um_per_px / 1000
    min_radius_px = int(min_radius_mm / pixel_size_mm)
    max_radius_px = int(max_radius_mm / pixel_size_mm)

    print(f"Searching for circles with radius: {min_radius_px}-{max_radius_px} pixels")
    print(f"Physical radius range: {min_radius_mm}-{max_radius_mm} mm")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(data_norm, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # Accumulator resolution = image resolution
        minDist=min_radius_px * 2,  # Minimum distance between circle centers
        param1=50,  # Upper threshold for Canny edge detector
        param2=20,  # Accumulator threshold (lower = more detections)
        minRadius=min_radius_px,
        maxRadius=max_radius_px)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Found {len(circles)} circle(s)")

        # Select the most central circle (closest to image center)
        img_center = np.array([height_data.shape[1] // 2, height_data.shape[0] // 2])
        distances = [np.linalg.norm(np.array([x, y]) - img_center) for x, y, r in circles]
        best_circle = circles[np.argmin(distances)]

        center_x, center_y, radius_px = best_circle
        radius_mm = radius_px * pixel_size_mm

        print(f"Best detection:")
        print(
            f"Center: ({center_x}, {center_y}) pixels = ({center_x * pixel_size_mm:.3f}, {center_y * pixel_size_mm:.3f}) mm")
        print(f"Radius: {radius_px} pixels = {radius_mm:.3f} mm")

        return center_x, center_y, radius_px, radius_mm, circles
    else:
        print("No circles detected!")
        return None, None, None, None, None


# Usage with your cropped data
center_x, center_y, radius_px, radius_mm, all_circles = detect_indentation_opencv(
    height_nachher_center,
    min_radius_mm=0.01,
    max_radius_mm=0.6,
    sensor_resolution_um_per_px=2.8334)

#x_center, y_center, radius_center = for_one_file.unexpected_input( scan_data,height_nachher)





def computeIndent(data_raw, radius_dist,center_x,center_y):
    angles = np.deg2rad(np.arange(1, 361))

    # Radial distances from 0 to radius_dist in 0.5 steps
    radii = np.arange(0, radius_dist + 0.01, 0.5)  # add small epsilon to include radius_dist

    # Shape: (num_radii, 360)
    cos_part = np.cos(angles)[None, :] * radii[:, None]
    sin_part = np.sin(angles)[None, :] * radii[:, None]

    # Compute full radius_vector: interleaved y (cos) and x (sin) components
    y_coords = center_y + cos_part
    x_coords = center_x + sin_part

    # radius_vector= np.vstack([y_coords.ravel(), x_coords.ravel()])
    # Interleave y and x columns into a single array: (num_radii, 720)
    radius_vector = np.empty((radii.size, 2 * angles.size))
    radius_vector[:, 0::2] = y_coords  # odd columns (1-based)
    radius_vector[:, 1::2] = x_coords  # even columns (1-based)
    return radius_vector  # , y_coords.shape

cut_vector= computeIndent(data_raw,90,center_x,center_y)


def two_dim_profile(radial_vector):
    cut_vector_y = radial_vector[:, 0::2]
    cut_vector_x = radial_vector[:, 1::2]

    from scipy.interpolate import RegularGridInterpolator
    # Better naming: makes it clear
    rows = np.arange(data_raw.shape[0])  # Y
    cols = np.arange(data_raw.shape[1])  # X
    interpolator = RegularGridInterpolator((rows, cols), data_raw)

    # Stack all (x, y) pairs across all angles: shape (radii, angles, 2)
    coords_all = np.stack((cut_vector_x, cut_vector_y), axis=-1)  # (radii, angles, 2)

    # Reshape into (N, 2) for interpolation
    coords_flat = coords_all.reshape(-1, 2)

    # Interpolate all at once
    values_flat = interpolator(coords_flat)  # shape: (radii * angles,)

    # Reshape back to (radii, angles)
    # This contains all 360 profiles
    profiles = values_flat.reshape(cut_vector_y.shape)

    # checks if each value is nan
    is_profile_nan = np.isnan(profiles)

    # Gives a percentage confidence of number of profiles that are nan
    # Example: When 1 then all 360 points are nan, other wise 0.5
    confidence_profile_nan = np.sum(is_profile_nan, axis=1) / is_profile_nan.shape[1]

    import data_preprocessing
    # Need inpaint here to better deal with the filtered profiles
    painted_output_profile = data_preprocessing.inpaint_nans_2(profiles)

    output_profile_2d = np.mean(painted_output_profile, axis=1)
    # Condition if 50% of measurements are nan then it is nan
    output_profile_2d[confidence_profile_nan > 0.5] = np.nan

    return output_profile_2d


# Get the 2D profile of the indent
output_profile_y = two_dim_profile(cut_vector)

punkt_abstand_x = 2.833
radial_distance_x = punkt_abstand_x * np.arange(0, radius_dist + 0.01, 0.5)

# Get the
rows_grid_shape, cols_grid_shape = np.shape(data_raw)
R, C = np.meshgrid(np.arange(0, cols_grid_shape), np.arange(0, rows_grid_shape))

# Looks at the region where the minimum could lie
factor_circle_radius = 30  # Based on load
circle_radius = factor_circle_radius / punkt_abstand_x
mask = (R - center_y) ** 2 + (C - center_x) ** 2 < circle_radius ** 2

# Get a vector with the possible minimum values
minimum_vector = data_raw[mask]
minimum_vector_sorted = np.sort(minimum_vector)

# Taking the 5th percentile of the sorted vector to be the minimum
minimum_value = np.nanpercentile(minimum_vector_sorted, 5)
output_profile_y[0:9] = minimum_value

output_profile = np.vstack((radial_distance_x, output_profile_y)).T

np.save("output_profile_2d", output_profile)

import matplotlib.pyplot as plt

plt.plot(output_profile[:, 0], output_profile[:, 1])
