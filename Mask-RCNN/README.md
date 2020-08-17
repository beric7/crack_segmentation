# Mask-RCNN for crack dectection
This is network is implemented by tensorflow. To start, install python packages in `requirement.txt`. You may hvae to download the corresponding version to run the code. There is sample output file from ARC under `/sample_arc_result/`

## Installation
Download `mask_rcnn_coco.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
Link: https://github.com/matterport/Mask_RCNN/releases

## crack.py
You need specify `ROOT_DIR`, `COCO_WEIGHTS_PATH` and `DEFAULT_LOGS_DIR` in `crack.py`. 
Set up your dataset into two folders:
`/img/`: contains orginial images
`/mask/`: contains ground truth mask
Note: This is required because when you specify the datas `/path/to/dataset/`, these two folders are required unde it. For example, `/home/CFD/train/img` and `/home/CFD/train/mask`

There are many configurable parameter are available. To access it, use:
```python crack.py -h
```
Train a new model starting from pre-trained COCO weights
```python3 crack.py train --dataset=/path/to/crack/dataset --weights=coco
```
Resume training a model that you had trained earlier:
```python3 crack.py train --dataset=/path/to/crack/dataset --weights=last
```
Predict and apply splash using the weight:
```python3 crack.py splash --weights=/path/to/mask_rcnn/mask_rcnn_crack.h5 --image=<file name or URL>
```
## mask_train.sh
This is for ARC online training.
