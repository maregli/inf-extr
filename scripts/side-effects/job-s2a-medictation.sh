#!/bin/bash
 
#SBATCH --job-name=s2a_line-label_medictation
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/side-effects/logs/s2a-medictation-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/side-effects/logs/s2a-medictation-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/line-label/finetune.py \
    --model_name medbert-512 \
    --task_type class \
    --batch_size 16 \
    --lr 2e-5 \
    --num_epochs 16 \
    --data_version medication \
    --num_labels 2 \
    --oversample
echo "Job finished"
