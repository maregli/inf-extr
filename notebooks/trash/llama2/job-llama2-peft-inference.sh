#!/bin/bash
 
#SBATCH --job-name=llama2_peft_inference
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/logs/peft-inference-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/logs/peft-inference-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=30G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/peft-inference.py \
    --job_id $SLURM_JOB_ID \
    --model_name llama2 \
    --peft_model_names ms-diag_llama2_bfloat16_lora_augmented_256 \
                        ms-diag_llama2_bfloat16_lora_augmented_512 \
                        ms-diag_llama2_bfloat16_prompt_augmented_256 \
                        ms-diag_llama2_bfloat16_prompt_augmented_512 \
                        ms-diag_llama2_bfloat16_ptune_augmented_256 \
                        ms-diag_llama2_bfloat16_ptune_augmented_512 \
    --quantization bfloat16 \
    --batch_size 4 \
    --split test \
    --num_labels 3
echo "Job finished"