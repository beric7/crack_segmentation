#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -t 03:30:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate rcnn

cd $PBS_O_WORKDIR

python crack.py train --dataset=/home/lxuan2/UNet/data/CFD/train_full_aug1/ --weights=coco -e 120

exit
