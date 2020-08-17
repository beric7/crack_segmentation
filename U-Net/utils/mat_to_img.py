# Convert .mat to image file for groundTruth
#
# Dependency:
# Author: Xuan Li
# Time: 7/07/2020

import os
from os import listdir
from os.path import splitext
from tqdm import tqdm
from glob import glob
import imageio
import numpy as np
from scipy.io import loadmat

dir_img = '/Users/xuanli/Desktop/CFD/mask/'
dir_out_img = '/Users/xuanli/Desktop/CFD/mask/'

if __name__ == "__main__":
    assert os.path.isdir(dir_img) , 'Input image directory does not exist'
    assert os.path.isdir(dir_out_img) , 'Output image directory does not exist'
    
    in_ids = [file for file in listdir(dir_img) if not file.startswith('.')]
    
    with tqdm(total=len(in_ids), desc=f'Image Processing', unit='img') as pbar:
        for id in in_ids:
            array = loadmat(dir_img+id)["groundTruth"][0][0][0]
            assert array.max() == 2 and array.min() == 1, f'hemo: array out of range: max {array.max()}, min {array.min()}'
            img = ((array - 1) * 255).astype(np.uint8)
            pathsplit = splitext(id)
            imageio.imwrite("{}.png".format(dir_out_img + pathsplit[0]), img)
            pbar.update(1)
