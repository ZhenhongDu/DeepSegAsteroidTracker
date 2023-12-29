from skimage.measure import label, regionprops, regionprops_table
from tifffile import imread, imwrite
import pandas as pd
import numpy as np
from utils import get_moving_stars, get_fixed_stars, fill_missing_frames, show_with_rect
from detectors import AsteroidTracker as tracker
import cv2
import math
import warnings
import time

warnings.filterwarnings('ignore')
track_colors = [(255, 255, 0), (127, 0, 255), (127, 0, 127),
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                ]
from collections import OrderedDict
# ================================
# Load images(raw_image and binary_image)
# ================================


# Diff SNR
# file_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_SNRs/'
# raw_file = file_path + 'simulate_data_SNR1.tif'
# file_name = file_path + 'simulate_data_SNR1_ResUnet2plus_seg.tif'
# save_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_SNRs/'


# Field function tracking
# file_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_field_crowding/'
# raw_file = file_path + 'simulate_data_star_num_100.tif'
# file_name = file_path + 'simulate_data_star_num_100_ResUnet2plus_seg.tif'
# save_path = '/mnt/data/asteroid/simulation_results/diff_field_crowding/'

file_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_field_crowding_2/'
raw_file = file_path + 'simulate_data_star_NUM_360.tif'
file_name = file_path + 'simulate_data_star_NUM_360_ResUnet2plus_seg.tif'
# save_path = '/mnt/data/asteroid/simulation_results/diff_field_crowding/'

# Multi-object tracking
# file_path = '/data/Data_zhenhong/asteroid/simulation_results/multi_asteroids/'
# raw_file = file_path + 'simulate_multi_asteroid_data1.tif'
# file_name = file_path + 'simulate_multi_asteroid_data1_ResUnet2plus_seg.tif'
# save_path = '/data/Data_zhenhong/asteroid/simulation_results/multi_asteroids/'

# diff speed tracking
# file_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_speeds/'
# raw_file = file_path + 'simulate_data_speed_27.tif'
# file_name = file_path + 'simulate_data_speed_27_ResUnet2plus_seg.tif'
# save_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_speeds/'

# Real asteroid tracking
# file_path = '/data/Data_zhenhong/asteroid/real_data/'
# raw_file = file_path + '0_6_full_size.tif'
# file_name = file_path + '0_6_full_size_crop_ResUNet2plus_seg1.tif'
# save_path = '/data/Data_zhenhong/asteroid/real_data/'

# =============================================

image = imread(file_name)[:,:,:]#[:-1, 100:-100, 100:-100]
raw_img = imread(raw_file).astype('float32')[:,:,:]#[:-1, 100:-100, 100:-100]
raw_img /= np.max(raw_img[:])
raw_img *= 255


start_time = time.time()
# ================================
# Obtain the connection area information
# from the binary input
# ================================
regions = []
star_nums = []

for m in range(image.shape[0]):
    # get the center coordinates of stars for each frame
    label_image = label(image[m])
    props = regionprops_table(label_image, properties=('centroid', 'bbox'))
    regions.append(props)
    # get the number of stars
    star_nums.append(len(props['centroid-0']))
# ================================
# get the fixed star coordinates
# calculate the index of images with
# the most stars OR manual selection
# ================================
METHOD = 'manual'
# show_with_rect(image[0], pd.DataFrame(regions[0]))
if METHOD == 'manual':
    num1 = 3
    num2 = 4
elif METHOD == 'auto':
    num1 = star_nums.index(sorted(star_nums)[-1])
    num2 = star_nums.index(sorted(star_nums)[-2])
else:
    raise Exception("Invalid METHOD! METHOD must be manual or auto")


fixed_star_coor = get_fixed_stars(regions[num1], regions[num2])
# show fixed stars
# show_with_rect(raw_img[1], fixed_star_coor)
# show_with_rect(image[1], fixed_star_coor)


# ================================
# tracking the asteroid
# ================================
tr = tracker()
# get the moving star coordinates
moving_stars = []

for n in range(image.shape[0]):
    moving_star = get_moving_stars(regions[n], fixed_star_coor, n)
    # show_with_rect(image[n], moving_star)

    moving_stars.append(moving_star)
    tr.update(moving_star)


tr.finish_mission()
# tr.check_slope()

det_asteroids = tr.asteroid

end_time = time.time()
print(end_time-start_time)


fill_frame_index = OrderedDict()

for id_num in det_asteroids.keys():
    det_asteroids[id_num], filled_index = fill_missing_frames(det_asteroids[id_num])
    fill_frame_index[id_num] = filled_index
# ================================
# Show the tracking result
# ================================


for i in range(len(raw_img)):
    # get current frame
    frame = raw_img[i].astype('uint8')
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # get asteroids coordinate
    asteroids_ids = det_asteroids.keys()

    for index, id in enumerate(asteroids_ids):
        asteroid = det_asteroids[id]
        asteroid_filled_frame_index = fill_frame_index[id]

        current_frame = list(asteroid['current_frame'])
        color = track_colors[index]

        if i in current_frame:
            if i in asteroid_filled_frame_index:
                color = (45, 182, 248)

            frame_index = current_frame.index(i)
            ## use centroid information
            half_width = 4
            start_point = (int(asteroid['centroid-1'][frame_index])-half_width, int(asteroid['centroid-0'][frame_index])-half_width)
            end_point = (int(asteroid['centroid-1'][frame_index])+half_width, int(asteroid['centroid-0'][frame_index])+half_width)

            # Line thickness of 2 px
            thickness = 2
            center_coordinates = (int(asteroid['centroid-1'][frame_index]), int(asteroid['centroid-0'][frame_index]))

            # Radius of circle
            radius = 10

            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)


    cv2.namedWindow('tracing', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tracing', (int(frame.shape[1] * 1), int(frame.shape[1] * 1)))

    # cv2.imwrite(save_path + "tracing_img{}.tif".format(i), frame[0:-1, 0:-1,:])
    roi_width = 80
    #cv2.imwrite(save_path + "tracing_img{}.tif".format(i),frame[int(asteroid['centroid-0'][i])-roi_width:int(asteroid['centroid-0'][i])+roi_width, int(asteroid['centroid-1'][i])-roi_width:int(asteroid['centroid-1'][i])+roi_width,:])
    # cv2.imwrite(save_path + "tracing_img{}.png".format(i), frame[50:-50,50:-50,:], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imshow('tracing', frame)

    # Slower the FPS
    cv2.waitKey(1000)


# When everything done, release the capture
cv2.destroyAllWindows()