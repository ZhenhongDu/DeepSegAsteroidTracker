from astropy.io import fits
from spalipy import Spalipy
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from tifffile import imread, imwrite


def _itk2np(x: sitk.Image):
    res = sitk.GetArrayFromImage(x)
    return res


def _np2itk(x):
    res = sitk.GetImageFromArray(x)
    return res


def diff_show(fixed, moving):
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(cimg))


file_path = 'E:/data/telescope/data2/enhanced_data/'
file_list = os.listdir(file_path)
# fixed_data = fits.getdata(file_path + file_list[12]).astype('float32')
fixed_data = imread(file_path + file_list[0]).astype('float32')
moving_data = []
for i in range(1, 10):
    # moving_data.append(fits.getdata(file_path + file_list[i]).astype('float32'))
    moving_data.append(imread(file_path + file_list[i]).astype('float32'))

print("Start registration!")
sp = Spalipy(moving_data, template_data=fixed_data)
sp.align()
out = sp.aligned_data

# diff_show(_np2itk(source_data.data), _np2itk(template_data.data))
# diff_show(_np2itk(out[0]), _np2itk(fixed_data.data))
# diff_show(_np2itk(out[5]), _np2itk(fixed_data))

save_path = 'E:/data/telescope/data2/registration_data/'
imwrite(save_path + 'star_regis_0.tif', fixed_data)
for i in range(1, 10):
    imwrite(save_path + 'star_regis_' + str(i) + '.tif', out[i-1])
