#!/bin/bash
 
#SBATCH --job-name=medbert_line-label_class_inference
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/line-label/medbert/logs/class-inference-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/line-label/medbert/logs/class-inference-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/line-label/inference.py \
    --model_name line-label_medbert-512_class \
    --task_type class \
    --split all \
    --batch_size 16
echo "Job finished"
