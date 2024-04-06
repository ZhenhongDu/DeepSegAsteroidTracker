import os
from skimage import io
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

img_path = 'E:/DZH-Results/Asteroid_tracking/Data/new_data/croped_patches/'
mask_path = 'E:/DZH-Results/Asteroid_tracking/Data/new_data/croped_patches/8bit/label/'
save_path = 'E:/DZH-Results/Asteroid_tracking/Data/new_data/dataset/'
mask_list = os.listdir(mask_path)

for mask_name in mask_list:
    if mask_name.endswith('_8bit.png'):
        mask = io.imread(os.path.join(mask_path, mask_name))
        img_name = mask_name.replace('_8bit.png', '.tif')
        img = io.imread(os.path.join(img_path, img_name))

        io.imsave(os.path.join(save_path,'image/',img_name), img)
        io.imsave(os.path.join(save_path,'mask/',mask_name.replace('_8bit.png', '_mask.tif')), mask)