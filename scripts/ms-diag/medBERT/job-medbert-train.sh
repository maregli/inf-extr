#!/bin/bash
 
#SBATCH --job-name=medbert_train
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/logs/train-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/logs/train-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx1080ti:1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/train.py \
    --quantization bfloat16 \
    --lr 0.0001 \
    --data original \
    --truncation_size 512 \
    --num_epochs 8
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/train.py \
    --quantization bfloat16 \
    --lr 0.0001 \
    --data zero-shot \
    --truncation_size 512 \
    --num_epochs 8
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/train.py \
    --quantization bfloat16 \
    --lr 0.0001 \
    --data augmented \
    --truncation_size 512 \
    --num_epochs 8
echo "Job finished"
