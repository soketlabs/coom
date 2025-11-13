#!/bin/bash
#SBATCH --job-name=eka_train_13_11_25
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu-long   # Replace with the correct partition name
#SBATCH --time=1-01:00:00
#SBATCH --nodelist=soketlab-node008,soketlab-node009
#SBATCH --output=logs/train_13_11_25_%x_%j.out
#SBATCH --error=logs/train_13_11_25_%x_%j.err



module load gcc
module load cuda
module load nccl
module load python/3.10
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run --devices 8 --accelerator gpu lightning_main.py 