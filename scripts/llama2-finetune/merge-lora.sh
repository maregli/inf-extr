#!/bin/bash
 
#SBATCH --job-name=merge_lora
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/llama2-finetune/logs/llama2-merge-lora-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/llama2-finetune/logs/llama2-merge-lora-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=30G
 
source ~/.bashrc

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/llama2-finetune/merge-lora.py \
    --model_name Llama2-MedTuned-13b \
    --peft_model_name Llama2-MedTuned-13b-LoRa \
    --new_model_name Llama2-MedTuned-13b-LoRa-merged 
echo "Job finished"
