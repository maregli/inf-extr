#!/bin/bash
 
#SBATCH --job-name=test-permission
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/logs/test-permission-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/logs/test-permission-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/test-permission.py
echo "Job finished"
