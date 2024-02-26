#!/bin/bash
 
#SBATCH --job-name=ms_medbert_train_oversample_all
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medbert/logs/all-oversample-train-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medbert/logs/all-oversample-train-train-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx1080ti:1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/finetune.py \
    --num_labels 4 \
    --data all \
    --data_augmentation oversample \
    --num_epochs 12 \
    --batch_size 8 \
    --lr 2e-4 
echo "Job finished"
