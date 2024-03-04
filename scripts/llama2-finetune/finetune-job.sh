#!/bin/bash
 
#SBATCH --job-name=llama2_finetune_unsupervised
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/llama2-finetune/logs/llama2-finetune-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/llama2-finetune/logs/llama2-finetune-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=15G
 
source ~/.bashrc

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/llama2-finetune/finetune.py \
    --model_name Llama2-MedTuned-13b \
    --new_model_name Llama2-MedTuned-13b-LoRa \
    --quantization "4bit" \
    --batch_size 16 \
    --lr 2e-4 \
    --num_epochs 1 \
    # --attn_implementation flash_attention_2 \
    # --bf16
echo "Job finished"
