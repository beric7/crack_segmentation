import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=512,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1], full_img.size[0])),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',help="Specify the file in which the model is stored", required=True)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help="Input folder or explicit input filenames", required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help="Output folder or explicit output filenames", required=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Preprocessed input image size",
                        default=512)

    return parser.parse_args()


def get_filenames(args):
    in_files = get_input_filenames(args)
    out_files = []

    if os.path.isdir(args.output[0]):
        for f in in_files:
            basename = os.path.splitext(os.path.basename(f))[0]
            out_files.append("{}_OUT.png".format(os.path.join(args.output[0], basename)))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output
        
    return in_files, out_files

def get_input_filenames(args):
    if os.path.isdir(args.input[0]):
        return [os.path.join(args.input[0], file) for file in os.listdir(args.input[0]) if not file.startswith('.')]
    else:
        for f in args.input:
            os.path.isfile(f)
        return args.input

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files, out_files = get_filenames(args)

    net = UNet(n_channels=3, n_classes=1, bilinear=False)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))
