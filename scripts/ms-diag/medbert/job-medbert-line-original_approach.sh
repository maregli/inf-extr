#!/bin/bash
 
#SBATCH --job-name=ms_medbert_train_line_original
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medbert/logs/line-original-train-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medbert/logs/line-original-train-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/finetune.py \
    --num_labels 3 \
    --data line \
    --data_augmentation original_approach \
    --num_epochs 16 \
    --batch_size 16 \
    --lr 2e-4 
echo "Job finished"
