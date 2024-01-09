#!/bin/bash
 
#SBATCH --job-name=llama2_lora
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/logs/lora-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/logs/lora-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=30G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/ \
    --model_name llama2 \
    --quantization bfloat16 \
    --peft_type ptune \
    --truncation_size 512 \
    --batch_size 8 \
    --lr 0.001 \
    --num_epochs 4 \
    --data augmented
echo "Job finished"