import os
import argparse
import torch
from tqdm import tqdm


import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.unets as module_arch

from utils.parse_config import ConfigParser
from tifffile import imwrite

def main(config):
    logger = config.get_logger('test')
    save_path = config['save_path']

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)


    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # save sample images, or do something with output here
            save_name_raw = save_path + 'raw_img_{}.tif'.format(i)
            save_name_label = save_path + 'label_img_{}.tif'.format(i)
            save_name_predict = save_path + 'predict_img_{}.tif'.format(i)
            imwrite(save_name_raw, torch.squeeze(data).cpu().numpy())
            imwrite(save_name_label, torch.squeeze(target).cpu().numpy())
            imwrite(save_name_predict, torch.squeeze(output).cpu().numpy())


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./test_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='./saved/models/Star_SegNet/0725_151951/checkpoint-epoch60.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
