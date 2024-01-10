#!/bin/bash
 
#SBATCH --job-name=llama2_lora
#SBATCH --output=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/logs/peft-inference-%j.out
#SBATCH --error=/cluster/home/eglimar/inf-extr/scripts/ms-diag/llama2/logs/peft-inference-%j.err
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=30G
 
source ~/.bashrc
conda activate inf-extr

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Name of model to be used. Defaults to llama2. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--peft_model_names", nargs="+", type=str, help="List of PEFT models for which to perform inference. Must be saved in the path: paths.MODEL_PATH/model_name. Must be compatible with base model.")
    parser.add_argument("--quantization", type=str, default=QUANTIZATION, help="Quantization. Must be one of 4bit, bfloat16 or float16. Defaults to 4bit")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size. Defaults to 4")
    parser.add_argument("--split", type=str, default=SPLIT, help="Split. Must be one of train, validation or test. Defaults to train")
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS, help="Number of labels (classes) to predict. Defaults to 3. Must be compatible with base model.")

    args = parser.parse_args()


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