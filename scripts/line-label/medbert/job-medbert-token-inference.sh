#!/bin/bash
 
#SBATCH --job-name=medbert_line-label_inference
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/line_label/medbert/logs/token-inference-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/line_label/medbert/logs/token-inference-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/line_label/medbert/inference-token.py \
    --model_name line-label_medbert-512_token \
    --task_type token \
    --split all \
    --batch_size 8
echo "Job finished"
