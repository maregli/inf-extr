#!/bin/bash
 
#SBATCH --job-name=medbert_line-label_class
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/line-label/medbert/logs/class-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/line-label/medbert/logs/class-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/line-label/finetune.py \
    --model_name medbert-512 \
    --task_type class \
    --batch_size 16 \
    --lr 2e-5 \
    --num_epochs 16
echo "Job finished"
