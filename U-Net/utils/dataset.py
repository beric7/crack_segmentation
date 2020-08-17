from os.path import splitext
from os import listdir
import numpy as np
from scipy import ndimage
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

def widen_pixl(img, scale):
    hidx, widx = np.where(img > 0)
    result = np.zeros_like(img)
    end_range = scale+1
    for i in range(len(hidx)):
        result[hidx[i]-scale:hidx[i]+end_range, widx[i]-scale:widx[i]+end_range] = 1
    return result

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=512, dis_trans=False, widen=0):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.dis_trans_enabled = dis_trans
        self.widen = widen

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, widen=0):
        assert scale > 0, 'Scale is too small'
        pil_img = pil_img.resize((scale, scale))

        img_nd = np.array(pil_img)
        
        if img_nd.max() > 1:
            img_nd = img_nd / 255
            
        if widen > 0:
            img_nd = widen_pixl(img_nd, widen)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans
        
    @classmethod
    def adv_preprocess(cls, pil_img, scale):
        assert scale > 0, 'Scale is too small'
        pil_img = pil_img.resize((scale, scale))
        img_nd = np.array(pil_img)
        
        if img_nd.max() > 1:
            img_nd = np.where(img_nd < 128, 0.0, img_nd)
            img_nd = np.where(img_nd > 127, 1.0, img_nd)
        
        di = ndimage.distance_transform_edt(img_nd)
        do = ndimage.distance_transform_edt(np.logical_not(img_nd))

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
            di = np.expand_dims(di, axis=2)
            do = np.expand_dims(do, axis=2)
        
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        di_trans = di.transpose((2, 0, 1))
        do_trans = do.transpose((2, 0, 1))
        
        return img_trans, di_trans, do_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        
        if self.dis_trans_enabled:
            mask, di, do = self.adv_preprocess(mask, self.scale)
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'di': torch.from_numpy(di).type(torch.FloatTensor),
                'do': torch.from_numpy(do).type(torch.FloatTensor)
            }
        
        mask_n = self.preprocess(mask, self.scale)
        if self.widen > 0 :
            mask_nn = self.preprocess(mask, self.scale, self.widen)
            return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_n).type(torch.FloatTensor),
            'mask_widen': torch.from_numpy(mask_nn).type(torch.FloatTensor)
            }
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_n).type(torch.FloatTensor)
        }
        
