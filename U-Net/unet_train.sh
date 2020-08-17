#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate default

cd $PBS_O_WORKDIR
cd UNet

python train.py -e 10 -b 4 -v 10

exit