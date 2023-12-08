#!/bin/bash
 
#SBATCH --job-name=test
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/line-label_prediction/llama2_zero-shot/logs/test-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/line-label_prediction/llama2_zero-shot/logs/test-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G
 
source ~/.bashrc
conda activate inf-extr
 
echo "Python version:" $(python --version)
echo "Starting job..."
python llama2_zero-shot_train.py
echo "Job finished"