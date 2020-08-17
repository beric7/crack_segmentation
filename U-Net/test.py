import sys
import os

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from utils.dataset import BasicDataset
from utils.TimeManagement import TimeManager
from torch.utils.data import DataLoader, random_split

dir_img = 'data/CFD/test/'
dir_mask = 'data/CFD/mask/'

def test_net(net, device, batch_size=4, scale=512, threshold=0.5):

    dataset = BasicDataset(dir_img, dir_mask, 512, False, 5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    tm = TimeManager()
        
    val_score, precision, recall = eval_net(net, loader, device, threshold)
                    
    if net.n_classes > 1:
        print('Validation cross entropy:', val_score)
    else:
        print('Validation Dice Coeff:', val_score)
        print('Validation Precision:', precision)
        print('Validation Recall:', recall)
        
    tm.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=512,
                        help='Preprocessed image size before testing. Default: 512 * 512')
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Create net
    net = UNet(n_channels=3, n_classes=1, bilinear=False)

    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device=device)
    logging.info("Model loaded !")

    try:
        test_net(net=net,
                  batch_size=args.batchsize,
                  device=device,
                  scale=args.scale,
                  threshold=args.mask_threshold)
    except KeyboardInterrupt:
        logging.info('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
