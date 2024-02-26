#!/bin/bash
 
#SBATCH --job-name=medbert_line-label_token
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/line-label/medbert/logs/token-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/line-label/medbert/logs/token-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx1080ti:1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/line-label/finetune-token.py \
    --model_name medbert-512 \
    --task_type token \
    --batch_size 8 \
    --lr 2e-5 \
    --num_epochs 20
echo "Job finished"
