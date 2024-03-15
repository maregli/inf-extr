#!/bin/bash

#SBATCH --job-name=medication_llama2-Medtuned13b_prompting_array
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/medication/logs/prompting_array-%A_%a.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/medication/logs/prompting_array-%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=15G

source ~/.bashrc

# Array of number of examples to run
num_examples_array=(1 2 4 8 10)

# Get the number of examples for this task
num_examples=${num_examples_array[$SLURM_ARRAY_TASK_ID]}

echo "Starting job with ID $SLURM_JOB_ID, task ID $SLURM_ARRAY_TASK_ID with $num_examples examples..."
python /cluster/home/eglimar/inf-extr/scripts/medication/prompting-outlines.py \
    --model_name Llama2-MedTuned-13b \
    --quantization 4bit \
    --attn_implementation flash_attention_2 \
    --prompt_strategies "few_shot_instruction" \
    --batch_size 1 \
    --num_examples $num_examples

echo "Job finished"

# use this to submit the job: sbatch --array=0-4 job_array-fi.sh
