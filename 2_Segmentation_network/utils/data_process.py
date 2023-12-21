# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 10:59
# @Author  : Du,Zhenhong
# @File    : data_process.py
# @aim     : 
# @change  :
import os
import math
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff


# 定义一个测试集，每次迭代时通过coordinate来确定裁剪的块
class testset(Dataset):
    def __init__(self, name_list, coordinate_list, noise_img):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        #fn = self.images[index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']

        noise_patch = self.noise_img[init_h:end_h, init_w:end_w]
        noise_patch=torch.from_numpy(np.expand_dims(noise_patch, 0))
        #target = self.target[index]
        return noise_patch,single_coordinate

    def __len__(self):
        return len(self.name_list)


def preprocess_lessMemoryPadding(opt, noise_im):
    img_h = opt.img_h
    img_w = opt.img_w

    gap_h = opt.gap_h
    gap_w = opt.gap_w

    name_list = []
    coordinate_list={}

    whole_w = noise_im.shape[1]
    whole_h = noise_im.shape[0]

    for x in range(0, int((whole_h - img_h + gap_h) / gap_h)):
        for y in range(0, int((whole_w - img_w + gap_w) / gap_w)):

            single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0}
            init_h = gap_h * x
            end_h = gap_h * x + img_h
            init_w = gap_w * y
            end_w = gap_w * y + img_w

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w

            patch_name = 'x' + str(x) + '_y' + str(y)
            # train_raw.append(noise_patch1.transpose(1,2,0))
            name_list.append(patch_name)
            # print(' single_coordinate -----> ',single_coordinate)
            coordinate_list[patch_name] = single_coordinate

    return name_list, coordinate_list

def preprocess_lessMemoryNoTail_chooseOne (args, noise_im):
    img_h = args.img_h
    img_w = args.img_w

    gap_h = args.gap_h
    gap_w = args.gap_w

    cut_w = (img_w - gap_w) / 2
    cut_h = (img_h - gap_h) / 2

    print('noise_im shape -----> ',noise_im.shape)

    whole_w = noise_im.shape[1]
    whole_h = noise_im.shape[0]

    num_w = math.ceil((whole_w - img_w + gap_w) / gap_w)
    num_h = math.ceil((whole_h - img_h + gap_h) / gap_h)

    name_list = []
    coordinate_list = {}
    # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
    # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
    for x in range(0,num_h):
        for y in range(0,num_w):
            single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
            if x != (num_h - 1):
                init_h = gap_h * x
                end_h = gap_h * x + img_h
            elif x == (num_h - 1):
                init_h = whole_h - img_h
                end_h = whole_h

            if y != (num_w - 1):
                init_w = gap_w * y
                end_w = gap_w * y + img_w
            elif y == (num_w - 1):
                init_w = whole_w - img_w
                end_w = whole_w

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w

            if y == 0:
                single_coordinate['stack_start_w'] = y * gap_w
                single_coordinate['stack_end_w'] = y * gap_w + img_w - cut_w
                single_coordinate['patch_start_w'] = 0
                single_coordinate['patch_end_w'] = img_w - cut_w
            elif y == num_w - 1:
                single_coordinate['stack_start_w'] = whole_w - img_w + cut_w
                single_coordinate['stack_end_w'] = whole_w
                single_coordinate['patch_start_w'] = cut_w
                single_coordinate['patch_end_w'] = img_w
            else:
                single_coordinate['stack_start_w'] = y * gap_w + cut_w
                single_coordinate['stack_end_w'] = y * gap_w + img_w - cut_w
                single_coordinate['patch_start_w'] = cut_w
                single_coordinate['patch_end_w'] = img_w - cut_w

            if x == 0:
                single_coordinate['stack_start_h'] = x * gap_h
                single_coordinate['stack_end_h'] = x * gap_h + img_h - cut_h
                single_coordinate['patch_start_h'] = 0
                single_coordinate['patch_end_h'] = img_h - cut_h
            elif x == num_h - 1:
                single_coordinate['stack_start_h'] = whole_h - img_h + cut_h
                single_coordinate['stack_end_h'] = whole_h
                single_coordinate['patch_start_h'] = cut_h
                single_coordinate['patch_end_h'] = img_h
            else:
                single_coordinate['stack_start_h'] = x * gap_h + cut_h
                single_coordinate['stack_end_h'] = x * gap_h + img_h - cut_h
                single_coordinate['patch_start_h'] = cut_h
                single_coordinate['patch_end_h'] = img_h - cut_h

            patch_name = 'x' + str(x) + '_y' + str(y)
            name_list.append(patch_name)
            coordinate_list[patch_name] = single_coordinate

    return name_list, coordinate_list


def singlebatch_test_save(single_coordinate,output_image):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    aaaa = output_image[patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return aaaa,stack_start_w,stack_end_w,stack_start_h,stack_end_h


def multibatch_test_save(single_coordinate,id,output_image):
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w=int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w=int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w=int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    output_image_id = output_image[id]
    aaaa = output_image_id[patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return aaaa,stack_start_w,stack_end_w,stack_start_h,stack_end_h

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='E:/Results/20220223_vlsfm_deconv/crop_test/crop_image/', help="A folder containing files for testing")
    parser.add_argument('--pth_path', type=str, default='D:/code/dzh_code/DeepCAD-RT/dzh_pt/pth/202203121550/RCAN_40_Iter_0091.pth', help="pth file root path")

    parser.add_argument('--img_w', type=int, default=216, help="the width of image sequence")
    parser.add_argument('--img_h', type=int, default=216, help="the height of image sequence")

    parser.add_argument('--gap_w', type=int, default=30, help='the width of image gap')
    parser.add_argument('--gap_h', type=int, default=30, help='the height of image gap')

    parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')

    parser.add_argument('--use_cuda', type=bool, default=True, help="the index of GPU you will use for computation")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")

    parser.add_argument('--output_dir', type=str, default='./results', help="output directory")

    opt = parser.parse_args()
    print('\033[1;31mTesting model -----> \033[0m')
    print(opt)
    return opt

if __name__ == '__main__':
    img = tiff.imread('K:/Results/result_dzh/210819_HE_stain/crop/210819_HE_stain_crop1_after_transform.tif')
    opt = get_argparse()
    n_list, cor_list = preprocess_lessMemoryPadding(opt, img[50])
    print(len(n_list))