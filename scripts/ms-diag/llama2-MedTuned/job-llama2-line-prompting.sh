#!/bin/bash
 
#SBATCH --job-name=ms_llama2-Medtuned_prompting
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/prompting-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/prompting-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=15G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/prompting.py \
    --model_name Llama2-MedTuned-7b \
    --quantization 4bit \
    --attn_implementation flash_attention_2 \
    --prompt_strategies zero_shot_vanilla zero_shot_instruction few_shot_vanilla few_shot_instruction \
    --data line \
    --split all \
    --batch_size 4
echo "Job finished"
