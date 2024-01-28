#!/bin/bash
 
#SBATCH --job-name=ms_llama2-Medtuned_prompting
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/prompting-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/prompting-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=15G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/prompting.py \
    --model_name Llama2-MedTuned-13b \
    --quantization 4bit \
    --data all_first_line_last \
    --split all \
    --batch_size 8 
echo "Job finished"
