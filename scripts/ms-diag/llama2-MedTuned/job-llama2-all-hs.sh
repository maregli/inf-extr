#!/bin/bash
 
#SBATCH --job-name=ms_llama2-Medtuned_hidden_state
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/hidden_state-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2-MedTuned/logs/hidden_state-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=15G
 
source ~/.bashrc
conda activate inf-extr

echo "Starting job with ID $SLURM_JOB_ID..."
python /cluster/home/eglimar/inf-extr/scripts/ms-diag/hidden_state_inference.py \
    --model_name Llama2-MedTuned-13b \
    --quantization 4bit \
    --results_files ms-diag_Llama2-MedTuned-13b_4bit_all_all_few_shot_instruction.pt \
                   ms-diag_Llama2-MedTuned-13b_4bit_all_all_few_shot_vanilla \
                   ms-diag_Llama2-MedTuned-13b_4bit_all_all_two_steps \
                   ms-diag_Llama2-MedTuned-13b_4bit_all_all_zero_shot_instruction \
                   ms-diag_Llama2-MedTuned-13b_4bit_all_all_zero_shot_vanilla
echo "Job finished"
