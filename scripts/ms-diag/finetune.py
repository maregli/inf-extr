
import torch

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import paths
from src.utils import (load_model_and_tokenizer, 
                       load_ms_data, 
                       prepare_ms_data, 
                       check_gpu_memory, 
                        get_optimizer_and_scheduler,
)

import argparse

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from peft import get_peft_config, get_peft_model

import evaluate

import numpy as np

import json

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description="Zero Shot Classification with Llama2-Chat")
    parser.add_argument("--job_id", type=str, default="unknown", help="Job ID")
    parser.add_argument("--model_name", type=str, default="medbert-512", help="Name of base model to be used. Defaults to medbert. Must be saved in the path: paths.MODEL_PATH/model_name")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization. Must be one of 4bit, bfloat16, float16 or None. Defaults to None")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size. Defaults to 4")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Defaults to 2e-4")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of Epochs. Defaults to 4")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation Steps. Defaults to 1.")
    parser.add_argument("--task_type", type=str, default="class", help="Task Type. Must be one of class or clm. Defaults to class")
    parser.add_argument("--data", type=str, default="line", help="Data. Must be one of line or all. Whether dataset consisting of single lines should be used or all text per rid.")
    parser.add_argument("--data_augmentation", type=str, default=None, help="Must be one of None, zero-shot, augmented original_approach. Defaults to None.")
    parser.add_argument("--num_labels", type=int, default=3, help="Number of Labels. Defaults to 3")
    parser.add_argument("--peft_config", type=str, default=None, help="PEFT Config. JSON-formatted configuration. Defaults to None")
    parser.add_argument("--attn_implementation", type=str, default=None, help="To implement Flash Attention 2 provide flash_attention_2. Defaults to None.")

    args = parser.parse_args()

    # Print or log the parsed arguments
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


seqeval = evaluate.load(os.path.join(paths.METRICS_PATH,"seqeval"))

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='macro')
    acc = accuracy_score(y_true=labels, y_pred=predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

    

def main():

    args = parse_args()
    JOB_ID = args.job_id
    MODEL_NAME = args.model_name
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    TASK_TYPE = args.task_type
    DATA = args.data
    DATA_AUGMENTATION = args.data_augmentation
    NUM_LABELS = args.num_labels
    PEFT_CONFIG = args.peft_config
    ATTN_IMPLEMENTATION = args.attn_implementation

    # Check GPU Memory
    check_gpu_memory()

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME, 
                                                quantization=QUANTIZATION, 
                                                num_labels=NUM_LABELS,
                                                attn_implementation=ATTN_IMPLEMENTATION,
                                                task_type=TASK_TYPE)
    
    # Get PEFT Config
    if PEFT_CONFIG is not None:
        config = json.loads(PEFT_CONFIG)
        PEFT_CONFIG = get_peft_config(config)
        model = get_peft_model(model, PEFT_CONFIG)
    else:
        config = {}

    print("Loaded Model and Tokenizer")

    # Load Data
    df = load_ms_data(data=DATA)

    # Prepare Data
    encoded_dataset = prepare_ms_data(df, tokenizer=tokenizer, data_augmentation = DATA_AUGMENTATION)

    print("Loaded Data")

    arg_names = [MODEL_NAME, QUANTIZATION, config.get("peft_type", None), TASK_TYPE, DATA, DATA_AUGMENTATION]
    arg_names = [arg_name for arg_name in arg_names if arg_name is not None]

    finetuned_model_name = f"ms-diag_{'_'.join(arg_names)}"

    
    # Num of training steps
    num_training_steps = len(encoded_dataset["train"]) // BATCH_SIZE * NUM_EPOCHS

    # Get optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model=model, num_training_steps=num_training_steps, learning_rate=LEARNING_RATE)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(paths.MODEL_PATH, "training", finetuned_model_name),          
        num_train_epochs=NUM_EPOCHS,             
        per_device_train_batch_size=BATCH_SIZE,  
        per_device_eval_batch_size=BATCH_SIZE,   
        warmup_steps=0,                               
        logging_dir='./logs',            
        logging_steps=20,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        seed=42,
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),                        
        args=training_args,                  
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest", pad_to_multiple_of=8),
    )

    print("Starting Training")

    # Train
    trainer.train()

    print("Finished Training")

    # Save Model
    print("Saving Model at:", paths.MODEL_PATH/finetuned_model_name)
    trainer.save_model(paths.MODEL_PATH/finetuned_model_name)

    # Save Tokenizer
    print("Saving Tokenizer at:", paths.MODEL_PATH/finetuned_model_name)
    tokenizer.save_pretrained(paths.MODEL_PATH/finetuned_model_name)
    return

if __name__ == "__main__":
    main()


