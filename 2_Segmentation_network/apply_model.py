import os

import numpy as np
import torch
from tifffile import imread, imwrite
import model.ResUNet2plus as module_arch
import time
from utils.data_process import testset, preprocess_lessMemoryNoTail_chooseOne, multibatch_test_save, singlebatch_test_save
import argparse
from torch.utils.data import DataLoader


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_w', type=int, default=256, help="the width of image sequence")
    parser.add_argument('--img_h', type=int, default=256, help="the height of image sequence")

    parser.add_argument('--gap_w', type=int, default=224, help='the width of image gap')
    parser.add_argument('--gap_h', type=int, default=224, help='the height of image gap')

    opt = parser.parse_args()
    print('\033[1;31mTesting model -----> \033[0m')
    print(opt)
    return opt


def seg_all_datadir_file(file_path, prefix, model_checkpoint):
    file_lists = []
    for full_file_name in os.listdir(file_path):
        if full_file_name.startswith(prefix):
            file_lists.append(full_file_name)

    for input_file in file_lists:
        save_dir = file_path + '/seg/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file_name = save_dir + input_file.replace('.tif', '_ResUnet2plus_seg.tif')
        input_file_path = file_path + input_file
        seg_big_stack(input_file_path, save_file_name, model_checkpoint)
        print("Processing file: {}".format(input_file))


def seg_big_stack(file_name, save_name, model_checkpoint):
    opt = get_argparse()
    # load the image stack
    stack = imread(file_name).astype('float32')
    stack /= np.max(stack[:])
    stack_size = np.shape(stack)
    # load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = module_arch.ResUnetPlusPlus(in_channel=1)
    checkpoint = torch.load(model_checkpoint)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ### test all images
    name_list, coordinate_list = preprocess_lessMemoryNoTail_chooseOne(opt, stack[0])

    prev_time = time.time()
    time_start = time.time()

    img_res = np.zeros(stack_size, dtype=np.float32)

    for N in range(stack_size[0]):
        template_img = stack[N]
        test_data = testset(name_list, coordinate_list, template_img)
        testloader = DataLoader(test_data, batch_size=16, shuffle=False)

        for iteration, (input_patch, single_coordinate) in enumerate(testloader):
            input_patch = input_patch.cuda()
            output = model(input_patch.cuda())
            output = (output > 0.5).float()

            ### determine approximate time left
            batch_done = iteration + N * len(testloader)
            batch_left = stack_size[0] * len(testloader) - batch_done
            time_left_seconds = int(batch_left * (time.time() - prev_time))
            prev_time = time.time()

            if iteration % 1 == 0:
                time_end = time.time()
                time_cost = time_end - time_start
                print(
                    '\r[Slice %d/%d] [Patch %d/%d] [Time Cost: %.4f s] [ETA: %s s]     '
                    % (
                        N + 1,
                        stack_size[0],
                        iteration + 1,
                        len(testloader),
                        time_cost,
                        time_left_seconds
                    ), end=' ')

            output_image = output.squeeze().detach().cpu().numpy()

            if (output_image.ndim == 2):
                turn = 1
            else:
                turn = output_image.shape[0]

            if (turn > 1):
                for id in range(turn):
                    temp, stack_start_w, stack_end_w, stack_start_h, stack_end_h = multibatch_test_save(
                        single_coordinate, id, output_image)
                    img_res[N, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = temp

            else:
                temp, stack_start_w, stack_end_w, stack_start_h, stack_end_h = singlebatch_test_save(single_coordinate,
                                                                                                     output_image)
                img_res[N, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = temp

    output_stack = img_res.squeeze().astype(np.float32)
    # output_img = output_img1[0:raw_noise_img.shape[0],0:raw_noise_img.shape[1],0:raw_noise_img.shape[2]]
    imwrite(save_name, output_stack)
    

if __name__ == '__main__':
    ## Diff speeds
    # path = '/data/Data_zhenhong/asteroid/simulation_results/diff_speeds/'
    # file_name = path + 'simulate_data_speed_27.tif'
    # save_name = path + 'simulate_data_speed_27_ResUnet2plus_seg.tif'


    ## Different_field_crowding
    # path = '/data/Data_zhenhong/asteroid/simulation_results/diff_field_crowding_2/'
    # file_name = path + 'simulate_data_star_NUM_360.tif'
    # save_name = path + 'simulate_data_star_NUM_360_ResUnet2plus_seg.tif'


    ## ALl files in the dir
    path = '/data/Data_zhenhong/asteroid/simulation_results/multi_SNR/Intensity_29/Simulated_data/'

    model_checkpoint = '/mnt/dzh/pytorch_seg/saved/models/Star_SegNet_simulated_data/1128_194327/checkpoint-epoch70.pth'
    # model_checkpoint = 'D:/code/dzh_code/Asteroid_tracking/pytorch_seg/saved/models/Star_SegNet_simulated_data/1127_220747/checkpoint-epoch80.pth'
    # seg_big_stack(file_name, save_name, model_checkpoint)
    # segment_single_large_img(file_name, save_name)
    seg_all_datadir_file(path, 'simulate_data_SNR_at_Intensity_29_Num_', model_checkpoint)
