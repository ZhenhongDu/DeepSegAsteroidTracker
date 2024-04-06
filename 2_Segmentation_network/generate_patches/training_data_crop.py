# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 13:38
# @Author  : Du,Zhenhong
# @File    : generate_datasets.py
# @aim     :
# @change  :

from tifffile import imread, imwrite
import numpy as np
import random
from skimage import io


def img_clip(img, p_min, p_max):
    return np.clip(img, p_min, p_max)

def rand_flag():
    return np.random.randint(0, 2)


def generate_2d_patch(x, patch_size=(512, 512), patch_num=200):
    raw_patch = []
    y_size, x_size = x.shape

    for _ in range(patch_num):

        yt = random.randint(0, y_size-patch_size[1])
        xt = random.randint(0, x_size-patch_size[0])

        raw_crop = x[yt:yt+patch_size[1], xt:xt+patch_size[0]]

        raw_patch.append(raw_crop)

    return raw_patch


if __name__ == '__main__':
    img_path = 'E:/DZH-Results/Asteroid_tracking/Data/new_data/after_regis/slices/'
    save_path = 'E:/DZH-Results/Asteroid_tracking/Data/new_data/croped_patches/'

    img = imread(img_path+'ROI_crop_data_5.tif')
    img = img[:, :].astype('float32')
    img /= np.max(img[:])

    out = generate_2d_patch(img, patch_size=(256, 256), patch_num=20)# (135:173) #(10,60),(147,173)
    c = np.array(out)

    # imwrite('temp.tif', volume, imagej=True, resolution=(1. / 2.6755, 1. / 2.6755), metadata = {'spacing': 3.947368, 'unit': 'um', 'axes': 'ZYX'})
    for i in range(c.shape[0]):
        # imwrite(save_path+"Star_Image_"+str(i+0)+".tif", c[i])
        io.imsave(save_path+"Star_Image_"+str(i+1100)+".png", c[i])
