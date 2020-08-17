import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from dice_loss import dice_coeff

def eval_net(net, loader, device, threshold=0.5):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_precision = 0
    tot_recall = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > threshold).float()
                mask_widen = None
                if len(batch) > 2:
                    mask_widen = batch['mask_widen']
                    mask_widen = mask_widen.to(device=device, dtype=mask_type)
                dic, prec, recall = dice_coeff(pred, true_masks, mask_widen,True)
                tot += dic.item()
                tot_precision += prec.item()
                tot_recall += recall.item()
            pbar.update()

    net.train()
    return tot / n_val, tot_precision / n_val, tot_recall / n_val

def eval_net_quite(net, loader, device, threshold=0.5):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > threshold).float()
            tot += dice_coeff(pred, true_masks).item()

    net.train()
    return tot / n_val