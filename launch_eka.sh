module load gcc
module load cuda
module load nccl
module load python/3.10
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 lightning_main.py 