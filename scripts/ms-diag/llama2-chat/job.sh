#!/bin/bash
 
#SBATCH --job-name=llama2_zero-shot
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/logs/zero-shot-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/logs/zero-shot-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-chat/zero_shot.py --job_id $SLURM_JOB_ID --quantization "bfloat16" --batch_size 2
echo "Job finished"
