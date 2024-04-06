from tifffile import imread, imwrite
import numpy as np
import random
from skimage import io



def img_clip(img, p_min, p_max):
    return np.clip(img, p_min, p_max)

def rand_flag():
    return np.random.randint(0,2)


def generate_2d_patch(x, patch_size=(512,512), patch_num=200):
    raw_patch = []
    y_size, x_size = x.shape

    for _ in range(patch_num):

        yt = random.randint(0, y_size-patch_size[1])
        xt = random.randint(0, x_size-patch_size[0])

        raw_crop = x[yt:yt+patch_size[1], xt:xt+patch_size[0]]

        raw_patch.append(raw_crop)

    return raw_patch

def generate_paired_2d_patch_from_3d(x, mask, patch_size=(256,256), patch_num=50, img_adjust=False):
    raw_patch = []
    mask_path = []

    z_size, y_size, x_size = x.shape

    if img_adjust:
        for ii in range(z_size):
            x[ii,:,:] = img_clip(x[ii,:,:], 0.03, 0.06)

    for _ in range(patch_num):
        zt = random.randint(0, z_size-1)
        yt = random.randint(0, y_size-patch_size[1])
        xt = random.randint(0, x_size-patch_size[0])

        raw_crop = x[zt, yt:yt+patch_size[1], xt:xt+patch_size[0]]
        mask_crop = mask[zt, yt:yt+patch_size[1], xt:xt+patch_size[0]]

        non_zero_count = np.count_nonzero(mask_crop)

        if non_zero_count < 20:
            continue

        raw_patch.append(raw_crop)
        mask_path.append(mask_crop)

    return raw_patch, mask_path


if __name__ == '__main__':
    img_path = 'E:/DZH-Results/Asteroid_tracking/Data/simulate_data/data_pair/'
    mask_path = 'E:/DZH-Results/Asteroid_tracking/Data/simulate_data/data_pair/'

    save_img_path = 'E:/DZH-Results/Asteroid_tracking/Data/simulate_data/dataset/image/'
    save_mask_path = 'E:/DZH-Results/Asteroid_tracking/Data/simulate_data/dataset/mask/'

    img = imread(img_path + 'simulate_data2.tif')
    mask = imread(mask_path + 'binary_data2.tif')

    img = img[:, :, :].astype('float32')
    img /= np.max(img[:])


    out0, out1 = generate_paired_2d_patch_from_3d(img, mask, patch_size=(256, 256), patch_num=120)# (135:173) #(10,60),(147,173)
    c0 = np.array(out0)
    c1 = np.array(out1)

    for i in range(c1.shape[0]):
        # imwrite(save_path+"Star_Image_"+str(i+0)+".tif", c[i])
        io.imsave(save_img_path+"Star_Image_"+str(i+1200)+".tif", c0[i])
        io.imsave(save_mask_path+"Star_Image_"+str(i+1200)+"_mask.tif", c1[i])
