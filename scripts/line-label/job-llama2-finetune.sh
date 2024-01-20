#!/bin/bash
 
#SBATCH --job-name=llama2_line-label_token
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/line_label/llama2-MedTuned/logs/token-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/line_label/llama2-MedTuned/logs/token-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=30G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/line_label/medbert/finetune-token.py \
    --model_name Llama2-MedTuned-13b \
    --task_type token \
    --batch_size 4 \
    --quantization 4bit \
    --peft_config '{"peft_type":"LORA", "r":16, "lora_alpha":32, "task_type":"TOKEN_CLS", "lora_dropout": 0.1, "bias":"all"}' \
    --lr 2e-5 \
    --num_epochs 20
echo "Job finished"
