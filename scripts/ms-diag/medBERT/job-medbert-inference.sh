#!/bin/bash
 
#SBATCH --job-name=medbert_inference
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/logs/inference-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/logs/inference-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx1080ti:1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=10G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/inference.py \
    --model_name ms-diag_medbert_bfloat16_finetuned_original_512 \
    --quantization bfloat16 
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/inference.py \
    --model_name ms-diag_medbert_bfloat16_finetuned_augmented_512 \
    --quantization bfloat16 
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/medBERT/inference.py \
    --model_name ms-diag_medbert_bfloat16_finetuned_zero-shot_512 \
    --quantization bfloat16 
echo "Job finished"
