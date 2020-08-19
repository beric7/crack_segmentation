# U-Net for crack dectection
This is network is implemented by pytorch. To start, install python packages in `requirement.txt`. There is sample output file from ARC under `/sample_arc_result/`

## train.py
You need specify `dir_img`, `dir_mask` and `dir_checkpoint` in `train.py`. 

There are many configurable parameter are available. To access it, use:
```bash
python train.py -h
```
To train a new model (sample):
```bash
python train.py -e 10 -b 4 -v 10
```
To train a previous model (sample):
```bash
python train.py -e 10 -b 4 -v 10 -f /path/to/model.pth
```
## test.py
You need specify `dir_img` and `dir_mask`  in `test.py`. 

There are many configurable parameter are available. To access it, use:
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
To predict a model:
```bash
python predict.py -m /path/to/model.pth -o /path/to/output.png -i /path/to/input.jpg
```
## unet_train.sh
This is for ARC online training.
