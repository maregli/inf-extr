import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_model_and_tokenizer, 
                       check_gpu_memory, 
)

import argparse

import torch

from transformers import TrainingArguments

from trl import SFTTrainer

from peft import get_peft_config, get_peft_model

import json


from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Llama2-finetuning unsupervised")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="Llama2-MedTuned-13b", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--new_model_name", type=str, default=None, help="Directory to save the model. Defaults to model_name_finetuned")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16, float16 or None. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Defaults to 2e-4")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of Epochs. Defaults to 1")
    parser.add_argument("--peft_config", type=str, default=None, help="PEFT Config. JSON-formatted configuration. Defaults to None in which case the default config is used.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")
    parser.add_argument("--bf16", type=bool, default=False, help="Enable bf16 training. Defaults to False")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args
    

def main():

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    NEW_MODEL_NAME = args.new_model_name
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    PEFT_CONFIG = args.peft_config
    ATTN_IMPLEMENTATION = args.attn_implementation
    BF16 = args.bf16

    # Check GPU Memory
    check_gpu_memory()

    ##########################
    # Model and Tokenizer
    ##########################

    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, 
                                                quantization=QUANTIZATION, 
                                                attn_implementation=ATTN_IMPLEMENTATION,
                                                task_type="clm")
    
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1 # recommended for quantized models I think

    ##########################
    # Data
    ##########################

    dataset = Dataset.load_from_disk(os.path.join(paths.DATA_PATH_PREPROCESSED, "text-finetune/kisim_diagnoses"))

    ##########################
    # Specifiations
    ##########################

    # adapted from: https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = os.path.join(paths.MODEL_PATH, "results")

    # Number of training epochs
    num_train_epochs = NUM_EPOCHS

    # Enable fp16/bf16 training
    fp16 = False
    bf16 = BF16

    # Batch size per GPU for training
    per_device_train_batch_size = BATCH_SIZE

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = LEARNING_RATE

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 25

    # Log every X updates steps
    logging_steps = 25

    # Maximum sequence length to use
    max_seq_length = 128

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = True

    # Get PEFT Config
    if PEFT_CONFIG is not None:
        config = json.loads(PEFT_CONFIG)

    else:
        # LoRA attention dimension
        lora_r = 64

        # Alpha parameter for LoRA scaling
        lora_alpha = 16

        # Dropout probability for LoRA layers
        lora_dropout = 0.1

        config = {
            "peft_type": "LORA",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias":"none",
            "task_type":"CAUSAL_LM",
        }

    PEFT_CONFIG = get_peft_config(config)

    ##########################
    # Training
    ##########################


    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=PEFT_CONFIG,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    print("Starting Training")

    # Train
    trainer.train()

    print("Finished Training")

    ##########################
    # Saving
    ##########################

    if NEW_MODEL_NAME is None:
        NEW_MODEL_NAME = MODEL_NAME + "_finetuned"

    print("Saving Model at:", paths.MODEL_PATH/NEW_MODEL_NAME)
    trainer.save_model(paths.MODEL_PATH/NEW_MODEL_NAME)

    print("Saving Tokenizer at:", paths.MODEL_PATH/NEW_MODEL_NAME)
    tokenizer.save_pretrained(paths.MODEL_PATH/NEW_MODEL_NAME)

    print("Saving training logs at:", paths.MODEL_PATH/NEW_MODEL_NAME/"log_history.pt")
    torch.save(trainer.state.log_history, paths.MODEL_PATH/NEW_MODEL_NAME/"log_history.pt")
    return

if __name__ == "__main__":
    main()


