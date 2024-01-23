#!/bin/bash
 
#SBATCH --job-name=ms_llama2-Medtuned_train
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/train-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/train-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=15G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/finetune.py \
    --model_name Llama2-MedTuned-13b \
    --quantization 4bit \
    --num_epochs 8 \
    --data line \
    --data_augmentation oversample \
    --task_type class \
    --num_labels 4 \
    --peft_config '{"peft_type":"LORA","lora_alpha":16,"lora_dropout":0.1, "r":64, "bias":"none","task_type":"SEQ_CLS"}' \
    --batch_size 16 \
    --lr 2e-4 \
    --attn_implementation flash_attention_2
echo "Job finished"
