#!/bin/bash
 
#SBATCH --job-name=medbert_inference_line_original
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medbert/logs/original-line-inference-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medbert/logs/original-line-inference-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx1080ti:1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/inference.py \
    --model_name ms-diag_medbert-512_class_line_original_approach \
    --split test \
    --num_labels 3 \
    --data line \
    --data_augmentation original_approach \
    --output_hidden_states 
echo "Job finished"
