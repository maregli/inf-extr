#!/bin/bash
 
#SBATCH --job-name=medication_llama2-Medtuned13b_prompting
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/medication/logs/prompting-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/medication/logs/prompting-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=15G
 
source ~/.bashrc

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/medication/prompting-outlines.py \
    --model_name Llama2-MedTuned-13b \
    --quantization 4bit \
    --attn_implementation flash_attention_2 \
    --prompt_strategies "all" \
    --batch_size 1 \
    --num_examples 10
echo "Job finished"
