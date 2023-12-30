#!/bin/bash
 
#SBATCH --job-name=test
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/logs/test-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/logs/test-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
echo "Job finished"
