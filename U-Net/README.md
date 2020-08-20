# U-Net for crack dectection
This is network is implemented by **pytorch**. To start, install python packages in `requirement.txt`. There is a couple of sample pretrained models and ARC output files under `/sample_arc_result/` folder.

## train.py
You need manually specify directories: **dir_img**, **dir_mask** and **dir_checkpoint** in `train.py`. 

There are many configurable parameters. To access them, use:
```bash
python train.py -h
```
To train a new model with epoch 10, batch size 4, validation percentage 10%: 
```bash
python train.py -e 10 -b 4 -v 10
```
To train a previous model with epoch 10, batch size 4, validation percentage 10%:
```bash
python train.py -e 10 -b 4 -v 10 -f /path/to/model.pth
```
Note that validation set is randomly seperated from training set every time you run the script.

## test.py
You need manually specify directories: **dir_img** and **dir_mask**  in `test.py`. 

There are  many configurable parameter are available. To access it, use:
```bash
python test.py -h
```
To test a model:
```bash
python test.py -m /path/to/model.pth
```
## predict.py

There are many configurable parameter are available. To access it, use:
```bash
python predict.py -h
```
To predict a image with a given model:
```bash
python predict.py -m /path/to/model.pth -o /path/to/output.png/or/folder -i /path/to/input.jpg/or/folder
```
To predict all images in the directory with given model: ( I will provide this very quick)
```bash
python predict.py -m /path/to/model.pth -o /path/to/output/folder -i /path/to/input/folder
```
## unet_train.sh
This is for ARC online training.

## dice_loss.py 
(Supplementary file)
This script contains the function that calculates F1 score, precision(optional), recall(optional). 

Note that detected pixels which are no more than five pixels away from the manually labeled pixels are considered true positive pixel. This is controlled by `target_widen` parameter in `dice_coeff`.

## eval.py
(Supplementary file)
There are two functions: 
`eval_net`: used for test and `target_widen` is enabled. This also prints out log during runtime.
`eval_net_quite`: used during train and `target_widen` is not enabled. No log.

## utils/data_vis.py
(Supplementary file)
Help for visualize results for image

## utils/dataset.py
(Supplementary file)
A torch dataset class for train and test set.

## utils/TimeManager.py
(Supplementary file)
This is used for showing time cost.

## utils/data_augmentation.py
(Supplementary file)
This script is used to augment images for training. This is not used during running.

## utils/mat_to_img.py
(Supplementary file)
This script is used to convert CFD dataset ground truth .mat files into .png images. This is not used during running.
