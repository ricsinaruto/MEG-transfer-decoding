#!/bin/bash
#$ -q gpu8.q
#$ -l gpu=2
#$ -cwd 
#$ -j y
#$ -o output.log


module load Anaconda3
source activate /gpfs3/users/woolrich/yaq921/.conda/envs/main
module load cuda/10.1
module load cudnn/7.6.5.32-10.1

python training.py