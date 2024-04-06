from photutils.segmentation import detect_sources
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch


file_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_field_crowding_2/successed_data/'

ASTEROID_NUM = 210

gt_file_name = 'gt_data_star_NUM_{}.tif'.format(ASTEROID_NUM)

gt_stack = imread(file_path+gt_file_name)
gt_image = gt_stack[1]

# simulation parameters
width_y = 512
width_x = 512

seg_threshold = 50

seg_res = detect_sources(gt_image, seg_threshold, npixels=8)
binary_seg = seg_res.data.astype('bool')

star_area = np.sum(binary_seg)
star_perc = star_area / (width_y*width_x)

print(star_perc)



plt.figure()

norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(gt_image, cmap='gray')
plt.show()


plt.figure()
plt.imshow(seg_res.data.astype('bool'), cmap='gray')
plt.show()